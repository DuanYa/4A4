class PlayZone {
    constructor(scene, x, y, position = 'center') {
        this.scene = scene;
        this.x = x;
        this.y = y;
        this.position = position;
        this.cards = [];
        this.container = scene.add.container(x, y).setDepth(38);
        this.bg = scene.add.graphics();
        this.label = scene.add.text(0, 70, '', {
            fontSize: '20px', fontFamily: 'Microsoft YaHei', color: '#d6e8d9', fontStyle: 'bold'
        }).setOrigin(0.5);
        this.passText = scene.add.text(0, 0, 'PASS', {
            fontSize: '42px', fontFamily: 'Arial Black, Microsoft YaHei', color: '#b9d2c2', fontStyle: 'bold', stroke: '#102d22', strokeThickness: 5
        }).setOrigin(0.5).setVisible(false);
        this.container.add([this.bg, this.label, this.passText]);
        this.drawBg(360, 142);
    }

    drawBg(w = 360, h = 142) {
        this.bg.clear();
        this.bg.fillStyle(0x000000, 0.18);
        this.bg.fillRoundedRect(-w / 2, -h / 2, w, h, 22);
        this.bg.lineStyle(2, 0xffffff, 0.08);
        this.bg.strokeRoundedRect(-w / 2, -h / 2, w, h, 22);
    }

    setPosition(x, y) {
        this.x = x;
        this.y = y;
        this.container.setPosition(x, y);
    }

    clear() {
        this.cards.forEach(c => c.destroy());
        this.cards = [];
        this.label.setText('');
        this.passText.setVisible(false);
        this.container.setVisible(false);
    }

    showPass(playerName) {
        this.cards.forEach(c => c.destroy());
        this.cards = [];
        this.drawBg(230, 116);
        this.container.setVisible(true);
        this.passText.setVisible(true);
        this.label.setText(playerName ? playerName + ' 不出' : '不出');
    }

    showCards(playedCards, labelText, playerName) {
        this.cards.forEach(c => c.destroy());
        this.cards = [];
        this.passText.setVisible(false);
        if (!playedCards || !playedCards.length) {
            this.clear();
            return;
        }
        this.container.setVisible(true);
        const scale = playedCards.length > 8 ? 0.54 : 0.63;
        const w = 92 * scale;
        const spacing = Math.min(12, (330 - w) / Math.max(1, playedCards.length));
        const total = playedCards.length * w + (playedCards.length - 1) * spacing;
        const bgW = Math.max(230, Math.ceil(total + 20));
        const bgH = Math.max(132, Math.ceil(136 * scale + 52));
        this.drawBg(bgW, bgH);
        const start = this.container.x - total / 2 + w / 2;
        playedCards.forEach((data, i) => {
            this.cards.push(new Card(this.scene, start + i * (w + spacing), this.container.y - 10, data, scale));
        });
        this.label.setText((playerName || '玩家') + (labelText ? ' · ' + labelText : ''));
    }

    destroy() {
        this.cards.forEach(c => c.destroy());
        this.container.destroy();
    }
}
