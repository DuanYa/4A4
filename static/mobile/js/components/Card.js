const SUIT_SYMBOLS = { spade: '♠', heart: '♥', club: '♣', diamond: '♦' };
const SUIT_COLORS = { spade: '#171717', club: '#171717', heart: '#d12f2f', diamond: '#d12f2f' };

class Card {
    constructor(scene, x, y, cardData, scale = 1) {
        this.scene = scene;
        this.cardData = cardData;
        this.selected = false;
        this.scale = scale;
        this.w = 92 * scale;
        this.h = 128 * scale;
        this.container = scene.add.container(Math.round(x), Math.round(y));
        this.draw();
    }

    draw() {
        const s = this.scale;
        const g = this.scene.add.graphics();
        const isRedJoker = this.cardData.rank === 'RJ';
        const isBlackJoker = this.cardData.rank === 'BJ';
        const rank = isRedJoker ? '大王' : isBlackJoker ? '小王' : this.cardData.rank;
        const suit = SUIT_SYMBOLS[this.cardData.suit] || '';
        const color = isRedJoker || isBlackJoker ? '#ffffff' : (SUIT_COLORS[this.cardData.suit] || '#171717');
        const fill = isRedJoker ? 0xd94b4b : isBlackJoker ? 0x343434 : 0xfffbef;
        g.fillStyle(0x000000, 0.22);
        g.fillRoundedRect(-this.w / 2 + 4 * s, -this.h / 2 + 6 * s, this.w, this.h, 10 * s);
        g.fillStyle(fill, 1);
        g.fillRoundedRect(-this.w / 2, -this.h / 2, this.w, this.h, 10 * s);
        g.lineStyle(2 * s, 0xd8c58f, 1);
        g.strokeRoundedRect(-this.w / 2, -this.h / 2, this.w, this.h, 10 * s);
        const rankText = this.scene.add.text(-this.w / 2 + 12 * s, -this.h / 2 + 10 * s, rank, {
            fontSize: Math.floor(20 * s) + 'px', fontFamily: 'Arial Black, Microsoft YaHei', color, fontStyle: 'bold'
        }).setOrigin(0, 0);
        const suitText = this.scene.add.text(0, 20 * s, suit || rank, {
            fontSize: Math.floor((suit ? 42 : 24) * s) + 'px', fontFamily: 'Microsoft YaHei', color, fontStyle: 'bold'
        }).setOrigin(0.5);
        this.container.add([g, rankText, suitText]);
        this.container.setSize(this.w, this.h);
    }

    setInteractive(callback) {
        this.container.setInteractive(new Phaser.Geom.Rectangle(-this.w / 2, -this.h / 2, this.w, this.h), Phaser.Geom.Rectangle.Contains);
        this.container.on('pointerdown', pointer => callback && callback(this, pointer));
        return this;
    }

    setSelected(selected) {
        if (this.selected === selected) return;
        this.selected = selected;
        this.scene.tweens.add({
            targets: this.container,
            y: this.container.y + (selected ? -34 * this.scale : 34 * this.scale),
            scale: selected ? 1.06 : 1,
            duration: 110,
            ease: 'Back.easeOut'
        });
    }

    destroy() { this.container.destroy(); }
}

class CardBack {
    constructor(scene, x, y, scale = 1) {
        this.scene = scene;
        this.scale = scale;
        this.w = 64 * scale;
        this.h = 88 * scale;
        this.container = scene.add.container(Math.round(x), Math.round(y));
        const g = scene.add.graphics();
        g.fillStyle(0x1d395f, 1);
        g.fillRoundedRect(-this.w / 2, -this.h / 2, this.w, this.h, 8 * scale);
        g.lineStyle(2 * scale, 0x7ab1ff, 0.6);
        g.strokeRoundedRect(-this.w / 2, -this.h / 2, this.w, this.h, 8 * scale);
        g.lineStyle(2 * scale, 0xffffff, 0.15);
        g.strokeCircle(0, 0, 20 * scale);
        this.container.add(g);
    }
    destroy() { this.container.destroy(); }
}
