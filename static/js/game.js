/**
 * 游戏渲染与交互逻辑
 */
const SUIT_SYMBOLS = {
    spade: '♠', heart: '♥', club: '♣', diamond: '♦', joker: '🃏'
};
const SUIT_COLORS = {
    spade: 'black', heart: 'red', club: 'black', diamond: 'red'
};
const HAND_NAMES = {
    SINGLE: '单张', PAIR: '对子', STRAIGHT: '单龙',
    DOUBLE_STRAIGHT: '双龙', BOMB3: '炸', BOMB4: '轰',
    JOKER_BOMB: '双王', SI_YAO_SI: '四幺四'
};

let mySeat = -1;
let selectedIndices = new Set();
let gameState = null;

function setMySeat(s) {
    AppLogger.info('Game', '设置我的座位', { seat: s });
    mySeat = s;
}

function renderCard(card, small) {
    const div = document.createElement('div');
    let cls = 'card';
    if (card.rank === 'RJ') cls += ' joker-r';
    else if (card.rank === 'BJ') cls += ' joker-b';
    else cls += ' ' + (SUIT_COLORS[card.suit] || 'black');
    div.className = cls;

    if (card.rank === 'RJ') {
        div.innerHTML = '<span class="card-rank">大</span><span class="card-suit">王</span>';
    } else if (card.rank === 'BJ') {
        div.innerHTML = '<span class="card-rank">小</span><span class="card-suit">王</span>';
    } else {
        const sym = SUIT_SYMBOLS[card.suit] || '';
        div.innerHTML = `<span class="card-suit">${sym}</span><span class="card-rank">${card.rank}</span>`;
    }
    return div;
}

function renderBackCards(count, vertical) {
    let html = '';
    for (let i = 0; i < Math.min(count, 14); i++) {
        html += '<span class="card-back"></span>';
    }
    return html;
}

function seatToView(seat) {
    const diff = ((seat - mySeat) + 4) % 4;
    return diff;
}

function renderGameState(state) {
    AppLogger.info('Game', '渲染游戏状态', {
        phase: state.phase,
        current_player_seat: state.current_player_seat,
        my_seat: mySeat,
        hand_size: (state.players.find(p => p.seat === mySeat) || {}).hand_size,
        played_cards: (state.played_cards || []).map(c => c.rank)
    });
    gameState = state;
    const myPlayer = state.players.find(p => p.seat === mySeat);

    const viewMap = {};
    state.players.forEach(p => { viewMap[seatToView(p.seat)] = p; });

    // 对家(view=2), 左家(view=1), 右家(view=3)
    [1, 2, 3].forEach(v => {
        const p = viewMap[v];
        if (!p) return;
        const el = document.getElementById('pname' + v);
        const cnt = document.getElementById('pcount' + v);
        const cards = document.getElementById('pcards' + v);
        const label = el.parentElement;
        el.textContent = p.name + (p.finished ? ' ✓' : '');
        cnt.textContent = p.hand_size + '张';
        label.className = 'player-label' +
            (state.current_player_seat === p.seat ? ' active' : '') +
            (p.finished ? ' finished' : '');
        cards.innerHTML = renderBackCards(p.hand_size, v === 1 || v === 3);
    });

    // 自己
    const me = viewMap[0];
    if (me) {
        document.getElementById('pname0').textContent = me.name;
        document.getElementById('pcount0').textContent = me.hand_size + '张';
        const label0 = document.getElementById('pname0').parentElement;
        label0.className = 'player-label' +
            (state.current_player_seat === mySeat ? ' active' : '');
        renderMyHand(me.hand || []);
    }

    // 出牌区
    renderPlayedCards(state);

    // 按钮状态
    updateButtons(state);

    // 叉/点弹窗
    handleChaDialog(state);
    handleDianDialog(state);

    // 顶部信息
    document.getElementById('gLevel').textContent = state.level_rank;
    document.getElementById('gStage').textContent =
        state.on_stage_team === 0 ? 'A队' : 'B队';
    document.getElementById('gPhaseText').textContent =
        getPhaseText(state);
}

function getPhaseText(state) {
    const names = {
        waiting: '等待中', dealing: '发牌中', playing: '出牌中',
        cha_asking: '叉牌询问', dian_asking: '点牌询问',
        round_end: '本局结束', tribute: '进贡中'
    };
    let t = names[state.phase] || state.phase;
    if (state.phase === 'playing' || state.phase === 'cha_asking' || state.phase === 'dian_asking') {
        const cp = state.players.find(p => p.seat === state.current_player_seat);
        if (cp) t += ' - ' + cp.name + '的回合';
    }
    return t;
}

function renderMyHand(hand) {
    const container = document.getElementById('myHand');
    container.innerHTML = '';
    selectedIndices.clear();
    hand.forEach((card, idx) => {
        const el = renderCard(card);
        el.dataset.index = idx;
        el.addEventListener('click', () => toggleSelect(el, idx));
        container.appendChild(el);
    });
}

function toggleSelect(el, idx) {
    if (el.classList.contains('selected')) {
        el.classList.remove('selected');
        selectedIndices.delete(idx);
    } else {
        el.classList.add('selected');
        selectedIndices.add(idx);
    }
    AppLogger.info('Game', '更新选牌', { selected_indices: Array.from(selectedIndices) });
}

function renderPlayedCards(state) {
    const container = document.getElementById('playedCards');
    const label = document.getElementById('actionLabel');
    container.innerHTML = '';
    if (state.played_cards && state.played_cards.length > 0) {
        state.played_cards.forEach(c => {
            container.appendChild(renderCard(c, true));
        });
        const lp = state.players.find(p => p.seat === state.last_play_seat);
        const ht = state.last_hand_type;
        let txt = lp ? lp.name : '';
        if (ht) txt += ' - ' + (HAND_NAMES[ht.category] || ht.category);
        label.textContent = txt;
    } else if (state.is_free_play) {
        label.textContent = '自由出牌';
    } else {
        label.textContent = '';
    }
}

function updateButtons(state) {
    const btnPlay = document.getElementById('btnPlay');
    const btnPass = document.getElementById('btnPass');
    const isMyTurn = state.current_player_seat === mySeat
        && state.phase === 'playing';
    btnPlay.disabled = !isMyTurn;
    btnPass.disabled = !isMyTurn || state.is_free_play;
    AppLogger.info('Game', '更新操作按钮', {
        is_my_turn: isMyTurn,
        is_free_play: state.is_free_play,
        play_disabled: btnPlay.disabled,
        pass_disabled: btnPass.disabled
    });
}

function handleChaDialog(state) {
    const dlg = document.getElementById('chaDialog');
    if (state.cha_asking_seat === mySeat) {
        document.getElementById('chaMsg').textContent =
            `有人出了 ${state.cha_rank}，你有对${state.cha_rank}，是否叉牌？`;
        AppLogger.info('Game', '显示叉牌询问', { cha_rank: state.cha_rank });
        dlg.classList.remove('hidden');
    } else {
        dlg.classList.add('hidden');
    }
}

function handleDianDialog(state) {
    const dlg = document.getElementById('dianDialog');
    if (state.dian_asking_seat === mySeat) {
        document.getElementById('dianMsg').textContent =
            `有人叉了 ${state.cha_rank}，你有${state.cha_rank}，是否点牌？`;
        AppLogger.info('Game', '显示点牌询问', { cha_rank: state.cha_rank });
        dlg.classList.remove('hidden');
    } else {
        dlg.classList.add('hidden');
    }
}

function addLog(text, cls) {
    const log = document.getElementById('gameLog');
    const div = document.createElement('div');
    if (cls) div.className = cls;
    div.textContent = text;
    log.appendChild(div);
    log.scrollTop = log.scrollHeight;
}

function showResult(data) {
    const dlg = document.getElementById('resultDialog');
    const title = document.getElementById('resultTitle');
    const body = document.getElementById('resultBody');

    let winner = '平局';
    if (data.winner_team === 0) winner = 'A队获胜';
    else if (data.winner_team === 1) winner = 'B队获胜';

    let detail = '';
    for (let t = 0; t < 2; t++) {
        const tn = t === 0 ? 'A队' : 'B队';
        const r = data['team' + t + '_result'];
        const rn = r === 'quan_dong' ? '全洞' :
                   r === 'ban_dong' ? '半洞' : '未赢';
        detail += `<div>${tn}: ${rn}</div>`;
    }
    if (data.upgrade > 0)
        detail += `<div>升级 ${data.upgrade} 级</div>`;
    if (data.zhi_j)
        detail += `<div style="color:#e74c3c">直J！退回打3</div>`;
    if (data.zhi_a)
        detail += `<div style="color:#e74c3c">直A！退回打J</div>`;
    detail += `<div>A队级别: ${data.new_level['0']}  B队级别: ${data.new_level['1']}</div>`;

    title.textContent = winner;
    AppLogger.info('Game', '显示结算弹窗', { winner: winner, result: data });
    body.innerHTML = detail;
    dlg.classList.remove('hidden');
}
