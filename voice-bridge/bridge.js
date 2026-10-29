const { Client, GatewayIntentBits } = require('discord.js');
const { joinVoiceChannel, EndBehaviorType, createAudioPlayer, createAudioResource, AudioPlayerStatus, NoSubscriberBehavior } = require('@discordjs/voice');
const prism = require('prism-media');
const net = require('net');
const dotenv = require('dotenv');
const path = require('path');
const fs = require('fs');
const { Transform } = require('stream');
const { exec, execFile, spawn } = require('child_process');

dotenv.config({ path: path.join(__dirname, '..', '.env') });

const tempDir = path.join(__dirname, 'temp_audio');
if (!fs.existsSync(tempDir)) {
    fs.mkdirSync(tempDir);
}
// Clean temp directory on startup
try {
    fs.readdirSync(tempDir).forEach(f => fs.unlinkSync(path.join(tempDir, f)));
} catch(e) { console.error('Failed to clean temp dir', e); }

const TOKEN = process.env.DISCORD_TOKEN;
const GUILD_ID = process.env.GUILD_ID;
const VOICE_CHANNEL_ID = process.env.VOICE_CHANNEL_ID;
const IGNORE_USER_IDS = (process.env.IGNORE_USER_IDS || "").split(',').filter(x => x);
const BRIDGE_PORT = parseInt(process.env.BRIDGE_PORT || '47777', 10);
const DEBUG = process.env.DEBUG === 'true';

const ignoredUsers = new Set(IGNORE_USER_IDS);
const subscriptions = new Map();
const stats = {};
const decoderStats = {};

let audioQueue = [];
let currentTrack = null;
let currentStreamProcess = null;
let hasEmittedQueueEmpty = true;
let isCommandTransition = false;
let trackStartTime = 0;

function killStreamProcess() {
    if (currentStreamProcess) {
        try {
            currentStreamProcess.kill();
        } catch (e) {}
        currentStreamProcess = null;
    }
}

const player = createAudioPlayer({
    behaviors: {
        noSubscriber: NoSubscriberBehavior.Play,
    },
});

const ttsPlayer = createAudioPlayer();
let isTtsPlaying = false;
let duckingPause = false;

ttsPlayer.on(AudioPlayerStatus.Idle, () => {
    isTtsPlaying = false;
    const { getVoiceConnection } = require('@discordjs/voice');
    const connection = getVoiceConnection(GUILD_ID);
    if (connection) {
        connection.subscribe(player);
        if (duckingPause) {
            duckingPause = false;
            player.unpause();
        }
    }
});

function prefetchNext() {
    const nextTrack = audioQueue.find(t => !t.localFile && !t.isDownloading);
    if (!nextTrack) return;
    
    nextTrack.isDownloading = true;
    const tmpFile = path.join(tempDir, `track_${Date.now()}`);
    execFile('yt-dlp', ['--no-playlist', '-f', 'bestaudio', '-o', tmpFile, nextTrack.url], (err, stdout, stderr) => {
        if (!err && fs.existsSync(tmpFile)) {
            nextTrack.localFile = tmpFile;
            if (DEBUG) console.log(`[DEBUG] Prefetched: ${nextTrack.title}`);
            prefetchNext(); // cascade prefetch
        } else {
            console.warn(`[WARNING] Prefetch failed for ${nextTrack.title}:`, stderr);
            nextTrack.isDownloading = false;
        }
    });
}

function playNext(wasCommand = false, lastDuration = 0) {
    if (audioQueue.length === 0) {
        currentTrack = null;
        killStreamProcess();
        if (!hasEmittedQueueEmpty && !wasCommand) {
            hasEmittedQueueEmpty = true;
            if (DEBUG) console.log(`[DEBUG] Queue empty, emitting queue_empty event. Last duration: ${lastDuration}s`);
            tcpClient.write(JSON.stringify({ event: 'queue_empty', last_duration: lastDuration, player: 'idle' }) + '\n');
        }
        return;
    }
    
    const track = audioQueue.shift();
    currentTrack = track;
    trackStartTime = Date.now();
    
    if (DEBUG) console.log(`[DEBUG] Playing next track: ${track.title}`);

    if (audioQueue.length === 0) {
        hasEmittedQueueEmpty = true;
        if (DEBUG) console.log(`[DEBUG] Queue empty after shift, emitting early queue_empty event`);
        if (!wasCommand) {
            tcpClient.write(JSON.stringify({ event: 'queue_empty', last_duration: 0, player: 'playing' }) + '\n');
        }
    } else {
        hasEmittedQueueEmpty = false;
    }

    if (track.localFile && fs.existsSync(track.localFile)) {
        if (DEBUG) console.log(`[DEBUG] Playing from prefetched file: ${track.localFile}`);
        killStreamProcess();
        startLocalPlayback(track.localFile);
        prefetchNext();
    } else {
        if (DEBUG) console.log(`[DEBUG] Not prefetched. Streaming with yt-dlp: ${track.title}`);
        killStreamProcess();

        const child = spawn('yt-dlp', ['--no-playlist', '-f', 'bestaudio', '-o', '-', track.url], {
            stdio: ['ignore', 'pipe', 'pipe']
        });
        currentStreamProcess = child;

        child.on('error', (err) => {
            console.warn('[WARNING] yt-dlp stream spawn error:', err);
            isCommandTransition = true;
            killStreamProcess();
            player.stop(true);
        });

        child.stderr.on('data', (d) => {
            if (DEBUG) console.log(`[DEBUG yt-dlp stderr] ${d.toString().trim()}`);
        });

        const resource = createAudioResource(child.stdout);
        resource.playStream.on('error', (err) => {
            console.warn('[WARNING] Audio resource stream error:', err);
            isCommandTransition = true;
            killStreamProcess();
            player.stop(true);
        });

        if (currentTrack) {
            tcpClient.write(JSON.stringify({
                event: 'track_start',
                title: currentTrack.title,
                artist: currentTrack.artist,
                url: currentTrack.url || '',
                video_id: currentTrack.video_id || null
            }) + '\n');
        }
        player.play(resource);
        prefetchNext();
    }
}

function startLocalPlayback(filePath) {
    try {
        const resource = createAudioResource(filePath);
        if (currentTrack) {
            tcpClient.write(JSON.stringify({
                event: 'track_start',
                title: currentTrack.title,
                artist: currentTrack.artist,
                url: currentTrack.url || '',
                video_id: currentTrack.video_id || null
            }) + '\n');
        }
        player.play(resource);
    } catch (err) {
        console.warn('[WARNING] Audio resource creation failed:', err);
        isCommandTransition = true;
        player.stop(true);
    }
}

player.on(AudioPlayerStatus.Idle, () => {
    const playDuration = (Date.now() - trackStartTime) / 1000;
    const wasCommand = isCommandTransition;
    
    if (DEBUG) console.log(`[DEBUG] AudioPlayer transitioned to Idle. Played for ${playDuration.toFixed(2)}s. isCommandTransition: ${isCommandTransition}`);
    
    killStreamProcess();
    
    let allowGaplessRefill = true;
    if (!isCommandTransition && currentTrack && playDuration < 30) {
        console.warn(`[WARNING] Track died prematurely after ${playDuration.toFixed(2)}s. Suspected stream failure. Disabling immediate gapless refill.`);
        allowGaplessRefill = false;
    }

    if (!allowGaplessRefill && audioQueue.length === 0) {
        if (DEBUG) console.log('[DEBUG] Suppressing queue_empty event due to premature track death.');
        hasEmittedQueueEmpty = true; 
    } else {
        hasEmittedQueueEmpty = false; // Reset to ensure we always try to fetch next track
    }

    isCommandTransition = false;
    
    // Clean up local file to save space
    if (currentTrack && currentTrack.localFile) {
        try { fs.unlinkSync(currentTrack.localFile); } catch(e) {}
    }

    playNext(wasCommand, playDuration);
});

player.on('error', (error) => {
    const playDuration = (Date.now() - trackStartTime) / 1000;
    console.warn(`[WARNING] AudioPlayer Error after ${playDuration.toFixed(2)}s: ${error.message}`);
    killStreamProcess();
    isCommandTransition = true;
    player.stop(true);
});

if (DEBUG) {
    setInterval(() => {
        for (const [userId, bytes] of Object.entries(stats)) {
            if (bytes > 0) {
                console.log(`[DEBUG] pcm bytes out for ${userId} in last 5s: ${bytes}`);
                stats[userId] = 0;
            }
        }
        for (const [userId, bytes] of Object.entries(decoderStats)) {
            if (bytes > 0) {
                console.log(`[DEBUG] decoded 48k PCM bytes for ${userId} in last 5s: ${bytes}`);
                decoderStats[userId] = 0;
            }
        }
    }, 5000);
}

const client = new Client({
    intents: [GatewayIntentBits.Guilds, GatewayIntentBits.GuildVoiceStates]
});

let tcpClient = new net.Socket();
let tcpBuffer = '';

tcpClient.connect(BRIDGE_PORT, '127.0.0.1', () => {
    console.log('Connected to Python bridge');
    client.login(TOKEN);
});

tcpClient.on('error', (err) => {
    console.error('TCP Connection error:', err);
    process.exit(1);
});

tcpClient.on('close', () => {
    console.log('TCP Connection closed');
    process.exit(0);
});

tcpClient.on('data', (chunk) => {
    tcpBuffer += chunk.toString('utf8');
    let lines = tcpBuffer.split('\n');
    tcpBuffer = lines.pop(); 
    
    for (let line of lines) {
        if (!line.trim()) continue;
        try {
            const msg = JSON.parse(line);
            if (msg.cmd === 'queue') {
                if (!msg.url || typeof msg.url !== 'string' || !msg.url.trim()) {
                    if (DEBUG) console.log(`[DEBUG] Rejecting queue cmd: missing or invalid url`);
                    if (msg.uuid) {
                        tcpClient.write(JSON.stringify({ event: 'queue_ack', status: 'rejected', reason: 'invalid_url', uuid: msg.uuid }) + '\n');
                    }
                    continue;
                }
                
                if (DEBUG) console.log(`[DEBUG] Received queue cmd: ${msg.title}`);
                audioQueue.push({ url: msg.url, title: msg.title, artist: msg.artist, video_id: msg.video_id || null });
                if (msg.uuid) {
                    tcpClient.write(JSON.stringify({ event: 'queue_ack', status: 'ok', uuid: msg.uuid }) + '\n');
                }
                
                if (player.state.status === AudioPlayerStatus.Idle) {
                    playNext(0);
                } else {
                    prefetchNext(); // trigger prefetch if player is busy
                }
            } else if (msg.cmd === 'skip') {
                if (DEBUG) console.log('[DEBUG] Received skip cmd');
                isCommandTransition = true;
                hasEmittedQueueEmpty = false;
                killStreamProcess();
                player.stop(); 


            } else if (msg.cmd === 'back') {
                if (DEBUG) console.log('[DEBUG] Received back cmd (handled by Python)');
                if (msg.uuid) {
                    tcpClient.write(JSON.stringify({ event: 'error', message: 'back is handled by Python', uuid: msg.uuid }) + '\n');
                }
            } else if (msg.cmd === 'pause') {
                if (DEBUG) console.log('[DEBUG] Received pause cmd');
                duckingPause = false;
                player.pause();
            } else if (msg.cmd === 'resume') {
                if (DEBUG) console.log('[DEBUG] Received resume cmd');
                player.unpause();
            } else if (msg.cmd === 'stop') {
                if (DEBUG) console.log('[DEBUG] Received stop cmd');
                isCommandTransition = true;
                killStreamProcess();
                audioQueue = [];
                duckingPause = false;
                player.stop(true);
            } else if (msg.cmd === 'status') {
                const resp = {
                    event: 'status',
                    queue: audioQueue.length,
                    playing: currentTrack ? { title: currentTrack.title, artist: currentTrack.artist } : null,
                    paused: player.state.status === AudioPlayerStatus.Paused || player.state.status === AudioPlayerStatus.AutoPaused,
                    up_next: audioQueue.slice(0, 10).map(t => ({ title: t.title, artist: t.artist })),
                    history_count: 0,
                    uuid: msg.uuid
                };
                tcpClient.write(JSON.stringify(resp) + '\n');
            } else if (msg.cmd === 'speak') {
                if (DEBUG) console.log('[DEBUG] Received speak cmd');
                const pcmBytes = Buffer.from(msg.pcm_b64, 'base64');
                const readable = new (require('stream').Readable)({
                    read() {
                        this.push(pcmBytes);
                        this.push(null);
                    }
                });
                const { StreamType } = require('@discordjs/voice');
                const resource = createAudioResource(readable, { inputType: StreamType.Raw });
                
                if (player.state.status === AudioPlayerStatus.Playing) {
                    player.pause();
                    duckingPause = true;
                }
                
                isTtsPlaying = true;
                const { getVoiceConnection } = require('@discordjs/voice');
                const connection = getVoiceConnection(GUILD_ID);
                if (connection) {
                    connection.subscribe(ttsPlayer);
                }
                ttsPlayer.play(resource);
            }
        } catch (e) {
            console.error('[CRITICAL] TCP JSON parse error in bridge.js:', e);
        }
    }
});

class Downsampler extends Transform {
    constructor() {
        super();
        this.buffer = Buffer.alloc(0);
    }

    _transform(chunk, encoding, callback) {
        this.buffer = Buffer.concat([this.buffer, chunk]);
        
        const numTriplets = Math.floor(this.buffer.length / 12);
        if (numTriplets === 0) {
            callback();
            return;
        }

        const outBuffer = Buffer.alloc(numTriplets * 2);

        for (let i = 0; i < numTriplets; i++) {
            const offset = i * 12;
            const l0 = this.buffer.readInt16LE(offset);
            const r0 = this.buffer.readInt16LE(offset + 2);
            const l1 = this.buffer.readInt16LE(offset + 4);
            const r1 = this.buffer.readInt16LE(offset + 6);
            const l2 = this.buffer.readInt16LE(offset + 8);
            const r2 = this.buffer.readInt16LE(offset + 10);
            const mono = Math.max(-32768, Math.min(32767, Math.floor((l0 + r0 + l1 + r1 + l2 + r2) / 6)));
            outBuffer.writeInt16LE(mono, i * 2);
        }

        this.push(outBuffer);
        this.buffer = this.buffer.slice(numTriplets * 12);
        
        callback();
    }
}

client.on('clientReady', () => {
    console.log(`Node bridge logged in as ${client.user.tag}`);
    ignoredUsers.add(client.user.id);
    if (DEBUG) console.log(`[DEBUG] Ignored bot's own user ID: ${client.user.id} to prevent feedback loop`);
    
    if (DEBUG) {
        try {
            const voicePkg = JSON.parse(fs.readFileSync(path.join(__dirname, 'node_modules', '@discordjs', 'voice', 'package.json')));
            console.log(`[@discordjs/voice version] ${voicePkg.version}`);
        } catch (e) {
            console.log('Could not read @discordjs/voice version');
        }
    }

    const guild = client.guilds.cache.get(GUILD_ID);
    if (!guild) {
        console.error('Guild not found');
        process.exit(1);
    }
    
    const connection = joinVoiceChannel({
        channelId: VOICE_CHANNEL_ID,
        guildId: GUILD_ID,
        adapterCreator: guild.voiceAdapterCreator,
        selfDeaf: false,
        selfMute: false, 
    });
    
    connection.subscribe(player); 

    if (DEBUG) {
        connection.on('stateChange', (oldState, newState) => {
            console.log(`[DEBUG] Connection state changed from ${oldState.status} to ${newState.status}`);
        });
        connection.on('error', error => {
            console.error(`[DEBUG] Connection error: ${error.message}`);
        });
    }

    connection.receiver.speaking.on('end', (userId) => {
        if (DEBUG) console.log(`[DEBUG] speaking end (event): ${userId}`);
        const existing = subscriptions.get(userId);
        if (existing) {
            if (DEBUG) console.log(`[DEBUG] Cleaning up audio chain for ${userId} (fallback from speaking end)`);
            try {
                if (existing.audioStream) existing.audioStream.destroy();
                if (existing.decoder) existing.decoder.destroy();
                if (existing.resampler) existing.resampler.destroy();
            } catch (e) {}
            subscriptions.delete(userId);
        }
    });

    connection.receiver.speaking.on('start', (userId) => {
        if (DEBUG) console.log(`[DEBUG] speaking start: ${userId}`);
        if (ignoredUsers.has(userId)) return;
        
        let existing = subscriptions.get(userId);
        if (existing) {
            if (DEBUG) console.log(`[DEBUG] Found existing subscription for ${userId}, destroying old chain first.`);
            try {
                if (existing.audioStream) existing.audioStream.destroy();
                if (existing.decoder) existing.decoder.destroy();
                if (existing.resampler) existing.resampler.destroy();
            } catch (e) {}
            subscriptions.delete(userId);
        }
        
        if (DEBUG) console.log(`[DEBUG] receiver.subscribe called for: ${userId}`);
        
        const audioStream = connection.receiver.subscribe(userId, {
            end: {
                behavior: EndBehaviorType.AfterSilence,
                duration: 1000,
            },
        });
        
        if (DEBUG) {
            audioStream.once('data', () => console.log(`[DEBUG] audioStream first data emitted for: ${userId}`));
        }
        
        const decoder = new prism.opus.Decoder({ rate: 48000, channels: 2, frameSize: 960 });
        
        decoder.on('data', (chunk) => {
            if (DEBUG) {
                decoderStats[userId] = (decoderStats[userId] || 0) + chunk.length;
            }
        });
        
        const resampler = new Downsampler();

        subscriptions.set(userId, { audioStream, decoder, resampler });

        audioStream.on('error', (err) => {
            if (DEBUG) console.warn(`[audioStream error] User ${userId}: ${err.message}`);
        });
        decoder.on('error', (err) => {
            console.error(`[decoder error] User ${userId}: ${err.message}`);
        });
        resampler.on('error', (err) => {
            console.error(`[resampler error] User ${userId}: ${err.message}`);
        });

        audioStream.pipe(decoder).pipe(resampler);

        resampler.on('data', (chunk) => {
            if (DEBUG) {
                stats[userId] = (stats[userId] || 0) + chunk.length;
            }
            if (tcpClient.writable) {
                const header = JSON.stringify({ u: userId, n: chunk.length }) + '\n';
                tcpClient.write(header);
                tcpClient.write(chunk);
            }
        });

        const cleanup = () => {
            const current = subscriptions.get(userId);
            if (current && current.audioStream === audioStream) {
                if (DEBUG) console.log(`[DEBUG] Cleaning up audio chain for ${userId} (audioStream ended/closed)`);
                subscriptions.delete(userId);
                try {
                    decoder.destroy();
                    resampler.destroy();
                } catch (e) {}
            }
        };

        audioStream.once('end', cleanup);
        audioStream.once('close', cleanup);
    });
});

process.on('SIGTERM', () => {
    client.destroy();
    tcpClient.destroy();
    process.exit(0);
});
process.on('SIGINT', () => {
    client.destroy();
    tcpClient.destroy();
    process.exit(0);
});
