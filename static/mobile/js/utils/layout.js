const DESIGN_WIDTH = 1334;
const DESIGN_HEIGHT = 750;

class LayoutManager {
    constructor(scene) {
        this.scene = scene;
        this.width = DESIGN_WIDTH;
        this.height = DESIGN_HEIGHT;
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.safe = { left: 24, right: DESIGN_WIDTH - 24, top: 18, bottom: DESIGN_HEIGHT - 18 };
    }

    update() {
        this.width = DESIGN_WIDTH;
        this.height = DESIGN_HEIGHT;
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.safe = { left: 24, right: DESIGN_WIDTH - 24, top: 18, bottom: DESIGN_HEIGHT - 18 };
    }

    x(v) { return Math.round(v); }
    y(v) { return Math.round(v); }
    s(v) { return v; }
    toX(v) { return this.x(v); }
    toY(v) { return this.y(v); }
    cx() { return DESIGN_WIDTH / 2; }
    cy() { return DESIGN_HEIGHT / 2; }

    domRect(x, y, w, h) {
        const canvas = this.scene.game.canvas;
        const rect = canvas.getBoundingClientRect();
        const sx = rect.width / DESIGN_WIDTH;
        const sy = rect.height / DESIGN_HEIGHT;
        return {
            x: rect.left + x * sx,
            y: rect.top + y * sy,
            w: w * sx,
            h: h * sy
        };
    }
}

function makeLayout(scene) {
    scene.layout = new LayoutManager(scene);
    return scene.layout;
}
