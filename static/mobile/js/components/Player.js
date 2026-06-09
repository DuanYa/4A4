class PlayerArea {
    constructor(scene, x, y, position) {
        this.scene = scene;
        this.x = x;
        this.y = y;
        this.position = position;
        this.cards = [];
        this.container = scene.add.container(x, y);
        this.ring = scene.add.graphics();
        this.nameText = scene.add.text(0, 50, '', {
            fontSize: '19px', fontFamily: 'Microsoft YaHei', color: '#ffffff', fontStyle: 'bold'
        }).setOrigin(0.5);
        this.countText = scene.add.text(0, 76, '', {
            fontSize: '16px', fontFamily: 'Microsoft YaHei', color: '#d6e8d9'
        }).setOrigin(0.5);
        this.badgeText = scene.add.text(0, -3, '', {
            fontSize: '28px', fontFamily: 'Arial Black', color: '#ffffff'
        }).setOrigin(0.5);
        this.container.add([this.ring, this.badgeText, this.nameText, this.countText]);
    }

    updatePlayer(playerData, isActive) {
        const teamColor = playerData.seat % 2 === 0 ? 0x1f9b5f : 0xd08a2d;
        this.ring.clear();
        this.ring.fillStyle(0x000000, 0.28);
        this.ring.fillCircle(4, 4, 42);
        this.ring.fillStyle(teamColor, 0.98);
        this.ring.fillCircle(0, 0, 42);
        this.ring.lineStyle(isActive ? 5 : 3, isActive ? 0xffd66b : 0xffffff, isActive ? 1 : 0.25);
        this.ring.strokeCircle(0, 0, 42);
        this.badgeText.setText(playerData.seat % 2 === 0 ? 'A' : 'B');
        this.nameText.setText((playerData.name || '玩家') + (playerData.finished ? ' ✓' : ''));
        this.countText.setText(playerData.hand_size + '张');
        this.container.setAlpha(playerData.finished ? 0.55 : 1);
        this.updateCardBacks(playerData.hand_size);
    }

    updateCardBacks(count) {
        this.cards.forEach(c => c.destroy());
        this.cards = [];
        if (!count) return;
        const max = Math.min(count, this.position === 'top' ? 8 : 5);
        for (let i = 0; i < max; i++) {
            let x = this.x;
            let y = this.y;
            if (this.position === 'top') {
                x += (i - max / 2) * 13 + 7;
                y += 76;
            } else if (this.position === 'left') {
                x += 78 + i * 10;
                y += 0;
            } else if (this.position === 'right') {
                x -= 78 + i * 10;
                y += 0;
            }
            this.cards.push(new CardBack(this.scene, x, y, 0.46));
        }
    }

    destroy() {
        this.cards.forEach(c => c.destroy());
        this.container.destroy();
    }
}
