class MobileUI {
    static bg(scene) {
        const l = scene.layout || makeLayout(scene);
        const g = scene.add.graphics();
        g.fillGradientStyle(0x062016, 0x062016, 0x176437, 0x176437, 1);
        g.fillRect(0, 0, l.width, l.height);
        g.fillStyle(0xffffff, 0.04);
        g.fillCircle(l.x(667), l.y(360), l.s(310));
        g.fillCircle(l.x(120), l.y(650), l.s(150));
        g.fillCircle(l.x(1210), l.y(110), l.s(180));
        return g;
    }

    static panel(scene, x, y, w, h, alpha = 0.76) {
        const l = scene.layout;
        const g = scene.add.graphics();
        g.fillStyle(0x102d22, alpha);
        g.fillRoundedRect(l.x(x - w / 2), l.y(y - h / 2), l.s(w), l.s(h), l.s(24));
        g.lineStyle(Math.max(1, l.s(2)), 0xffd66b, 0.35);
        g.strokeRoundedRect(l.x(x - w / 2), l.y(y - h / 2), l.s(w), l.s(h), l.s(24));
        return g;
    }

    static button(scene, x, y, w, h, text, color, cb) {
        const l = scene.layout;
        const c = scene.add.container(l.x(x), l.y(y));
        const g = scene.add.graphics();
        const sw = l.s(w);
        const sh = l.s(h);
        const draw = (enabled = true, down = false) => {
            g.clear();
            g.fillStyle(enabled ? color : 0x66706b, enabled ? 1 : 0.55);
            g.fillRoundedRect(-sw / 2, -sh / 2, sw, sh, sh / 2);
            g.lineStyle(Math.max(1, l.s(3)), 0xffffff, enabled ? 0.22 : 0.08);
            g.strokeRoundedRect(-sw / 2, -sh / 2, sw, sh, sh / 2);
            c.setScale(down ? 0.96 : 1);
        };
        const label = scene.add.text(0, 0, text, {
            fontSize: Math.floor(l.s(h * 0.38)) + 'px',
            fontFamily: 'Microsoft YaHei',
            color: '#ffffff',
            fontStyle: 'bold'
        }).setOrigin(0.5);
        c.add([g, label]);
        c.text = label;
        const hit = scene.add.rectangle(0, 0, sw, sh, 0xffffff, 0).setOrigin(0.5);
        c.add(hit);
        c.enabled = true;
        c.isDown = false;
        c.setEnabled = enabled => { c.enabled = enabled; c.isDown = false; draw(enabled); label.setAlpha(enabled ? 1 : 0.55); };
        hit.setInteractive({ useHandCursor: true });
        hit.on('pointerdown', () => {
            if (!c.enabled || c.isDown) return;
            c.isDown = true;
            draw(true, true);
            label.setAlpha(0.86);
            if (navigator.vibrate) navigator.vibrate(8);
            if (cb) cb();
            scene.time.delayedCall(70, () => { c.isDown = false; label.setAlpha(c.enabled ? 1 : 0.55); draw(c.enabled); });
        });
        hit.on('pointerup', () => { c.isDown = false; label.setAlpha(c.enabled ? 1 : 0.55); draw(c.enabled); });
        hit.on('pointerout', () => { c.isDown = false; draw(c.enabled); });
        draw(true);
        return c;
    }

    static text(scene, x, y, value, size = 28, color = '#ffffff', style = '') {
        const l = scene.layout;
        return scene.add.text(l.x(x), l.y(y), value, {
            fontSize: Math.floor(l.s(size)) + 'px',
            fontFamily: 'Microsoft YaHei',
            color,
            fontStyle: style
        }).setOrigin(0.5);
    }

    static toast(scene, message, color = 0x000000) {
        const l = scene.layout;
        if (scene._toast) scene._toast.destroy();
        const c = scene.add.container(l.x(667), l.y(606)).setDepth(2000);
        const bg = scene.add.graphics();
        bg.fillStyle(color, 0.82);
        bg.fillRoundedRect(-l.s(260), -l.s(34), l.s(520), l.s(68), l.s(34));
        const t = scene.add.text(0, 0, message, {
            fontSize: Math.floor(l.s(24)) + 'px', fontFamily: 'Microsoft YaHei', color: '#ffffff'
        }).setOrigin(0.5);
        c.add([bg, t]);
        scene._toast = c;
        scene.tweens.add({ targets: c, y: l.y(566), alpha: 0, delay: 1200, duration: 300, onComplete: () => c.destroy() });
        return c;
    }

    static inputBox(scene, x, y, w, h, placeholder, maxLength = 12) {
        const l = scene.layout;
        const c = scene.add.container(l.x(x), l.y(y));
        const bg = scene.add.graphics();
        const sw = l.s(w);
        const sh = l.s(h);
        bg.fillStyle(0x081b14, 0.82);
        bg.fillRoundedRect(-sw / 2, -sh / 2, sw, sh, sh / 2);
        bg.lineStyle(Math.max(1, l.s(2)), 0xffd66b, 0.45);
        bg.strokeRoundedRect(-sw / 2, -sh / 2, sw, sh, sh / 2);
        const label = scene.add.text(0, 0, placeholder, {
            fontSize: Math.floor(l.s(22)) + 'px', fontFamily: 'Microsoft YaHei', color: '#b9d2c2'
        }).setOrigin(0.5);
        c.add([bg, label]);
        c.value = '';
        c.placeholder = placeholder;
        c.maxLength = maxLength;
        c.label = label;
        c.setSize(sw, sh).setInteractive(new Phaser.Geom.Rectangle(-sw / 2, -sh / 2, sw, sh), Phaser.Geom.Rectangle.Contains);
        c.bounds = l.domRect(x - w / 2, y - h / 2, w, h);
        c.on('pointerdown', () => {
            c.bounds = l.domRect(x - w / 2, y - h / 2, w, h);
            MobileUI.focusNativeInput(scene, c);
        });
        c.setValue = value => {
            c.value = String(value || '').slice(0, maxLength);
            label.setText(c.value || c.placeholder);
            label.setColor(c.value ? '#ffffff' : '#b9d2c2');
        };
        return c;
    }

    static focusNativeInput(scene, input) {
        let native = input.nativeInput;
        if (!native) {
            native = document.createElement('input');
            native.className = 'mobile-ime-input';
            native.autocapitalize = 'off';
            native.autocomplete = 'off';
            native.autocorrect = 'off';
            native.spellcheck = false;
            native.maxLength = input.maxLength;
            document.body.appendChild(native);
            native.addEventListener('input', () => input.setValue(native.value));
            native.addEventListener('blur', () => {
                native.style.width = '1px';
                native.style.height = '1px';
                native.style.left = '-100px';
                native.style.top = '-100px';
            });
            input.nativeInput = native;
        }
        const b = input.bounds;
        native.value = input.value || '';
        native.maxLength = input.maxLength;
        native.style.left = Math.round(b.x) + 'px';
        native.style.top = Math.round(b.y) + 'px';
        native.style.width = Math.round(b.w) + 'px';
        native.style.height = Math.round(b.h) + 'px';
        native.style.fontSize = Math.max(16, Math.round(b.h * 0.42)) + 'px';
        native.focus({ preventScroll: true });
        native.select();
    }

    static keyboard(scene, input) {
        const l = scene.layout;
        if (scene._keyboard) scene._keyboard.destroy();
        const c = scene.add.container(l.x(667), l.y(375)).setDepth(3000);
        const overlay = scene.add.rectangle(0, 0, l.width, l.height, 0x000000, 0.62).setOrigin(0.5);
        const panel = MobileUI.panel(scene, 667, 375, 980, 470, 0.96);
        panel.x -= l.x(667);
        panel.y -= l.y(375);
        const title = scene.add.text(0, -l.s(190), input.placeholder, {
            fontSize: Math.floor(l.s(26)) + 'px', fontFamily: 'Microsoft YaHei', color: '#ffd66b', fontStyle: 'bold'
        }).setOrigin(0.5);
        const valueText = scene.add.text(0, -l.s(145), input.value || '', {
            fontSize: Math.floor(l.s(26)) + 'px', fontFamily: 'Microsoft YaHei', color: '#ffffff'
        }).setOrigin(0.5);
        c.add([overlay, panel, title, valueText]);
        const keys = ['1','2','3','4','5','6','7','8','9','0','A','B','C','D','E','F','G','H','J','K','L','M','N','P','Q','R','S','T','W','X','Y','Z'];
        keys.forEach((key, idx) => {
            const col = idx % 8;
            const row = Math.floor(idx / 8);
            const bx = -l.s(350) + col * l.s(100);
            const by = -l.s(82) + row * l.s(62);
            const b = MobileUI.button(scene, 667 + bx / l.scale, 375 + by / l.scale, 78, 46, key, 0x244c3a, () => {
                if (input.value.length < input.maxLength) input.setValue(input.value + key);
                valueText.setText(input.value);
            });
            b.x -= l.x(667); b.y -= l.y(375);
            c.add(b);
        });
        const del = MobileUI.button(scene, 542, 563, 160, 52, '删除', 0x7f8c8d, () => {
            input.setValue(input.value.slice(0, -1));
            valueText.setText(input.value);
        });
        const ok = MobileUI.button(scene, 792, 563, 160, 52, '确定', 0xe6a800, () => {
            c.destroy();
            scene._keyboard = null;
        });
        del.x -= l.x(667); del.y -= l.y(375);
        ok.x -= l.x(667); ok.y -= l.y(375);
        c.add([del, ok]);
        scene._keyboard = c;
    }

    static destroyNativeInputs(scene) {
        if (!scene) return;
        ['nameInput', 'roomInput'].forEach(key => {
            const input = scene[key];
            if (input && input.nativeInput) {
                input.nativeInput.remove();
                input.nativeInput = null;
            }
        });
    }
}
