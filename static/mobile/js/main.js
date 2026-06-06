const config = {
    type: Phaser.AUTO,
    parent: 'game-container',
    width: DESIGN_WIDTH,
    height: DESIGN_HEIGHT,
    backgroundColor: '#061b13',
    render: {
        antialias: true,
        antialiasGL: true,
        pixelArt: false,
        roundPixels: true,
        powerPreference: 'high-performance'
    },
    resolution: Math.min(window.devicePixelRatio || 1, 3),
    scale: {
        mode: Phaser.Scale.FIT,
        autoCenter: Phaser.Scale.CENTER_BOTH,
        width: DESIGN_WIDTH,
        height: DESIGN_HEIGHT
    },
    input: { activePointers: 5 },
    scene: [LobbyScene, WaitingScene, GameScene],
    fps: { target: 60, forceSetTimeOut: false }
};

function updateOrientationHint() {
    const hint = document.getElementById('rotate-hint');
    if (!hint) return;
    const portrait = window.innerHeight > window.innerWidth;
    hint.classList.toggle('visible', portrait);
}

window.addEventListener('load', () => {
    AppLogger.info('Main', '初始化横屏移动端游戏', {
        viewport: { width: window.innerWidth, height: window.innerHeight },
        design: { width: DESIGN_WIDTH, height: DESIGN_HEIGHT },
        dpr: window.devicePixelRatio || 1
    });

    const params = new URLSearchParams(window.location.search);
    window.__shareRoomId = params.get('room') || null;
    if (window.__shareRoomId) {
        AppLogger.info('Main', '检测到分享房间链接', { roomId: window.__shareRoomId });
    }

    wsManager.connect();
    const game = new Phaser.Game(config);
    window.game = game;
    window.wsManager = wsManager;

    const loading = document.getElementById('loading');
    if (loading) loading.classList.add('hidden');
    updateOrientationHint();
    window.addEventListener('resize', updateOrientationHint);
    window.addEventListener('orientationchange', () => setTimeout(updateOrientationHint, 250));
});
