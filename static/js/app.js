/**
 * 应用入口 - 连接各模块
 */
(function() {
    const $lobby = document.getElementById('lobby');
    const $waiting = document.getElementById('waiting');
    const $game = document.getElementById('gameScreen');

    function showScreen(name) {
        AppLogger.info('App', '切换界面', { screen: name });
        [$lobby, $waiting, $game].forEach(s => s.classList.remove('active'));
        if (name === 'lobby') $lobby.classList.add('active');
        else if (name === 'waiting') $waiting.classList.add('active');
        else if (name === 'game') $game.classList.add('active');
    }

    let currentRoomId = '';

    // 连接WebSocket
    ws.connect();

    // 加入房间按钮
    document.getElementById('btnJoin').addEventListener('click', () => {
        const name = document.getElementById('playerName').value.trim() || '玩家';
        const rid = document.getElementById('roomId').value.trim();
        AppLogger.info('App', '点击加入/创建房间', { name: name, room_id: rid || '(auto)' });
        const msg = document.getElementById('lobbyMsg');

        if (rid) {
            ws.send('join_room', { room_id: rid, name: name });
        } else {
            fetch('/api/rooms', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({})
            })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    msg.textContent = data.error;
                    return;
                }
                currentRoomId = data.room_id;
                AppLogger.info('App', '房间创建成功', data);
                ws.send('join_room', { room_id: data.room_id, name: name });
            })
            .catch(e => { msg.textContent = '网络错误'; });
        }
    });

    // 事件处理
    ws.on('joined', data => {
        currentRoomId = data.room_id;
        AppLogger.info('App', '加入房间成功', data);
        setMySeat(data.seat);
        document.getElementById('roomIdDisplay').textContent = currentRoomId;
        showScreen('waiting');
        loadCheckpoints();
    });

    ws.on('error', data => {
        AppLogger.warn('App', '收到错误事件', data);
        const lobbyMsg = document.getElementById('lobbyMsg');
        if ($lobby.classList.contains('active')) {
            lobbyMsg.textContent = data.message;
        } else {
            addLog('错误: ' + data.message);
        }
    });

    ws.on('room_state', data => {
        AppLogger.info('App', '更新房间状态', {
            room_id: data.room_id,
            player_count: data.player_count,
            game_active: data.game_active
        });
        const slots = ['slot0', 'slot1', 'slot2', 'slot3'];
        data.players.forEach((p, i) => {
            const el = document.getElementById(slots[i]);
            const nameEl = el.querySelector('.seat-name');
            if (p) {
                nameEl.textContent = p.name;
                el.classList.add('occupied');
                el.classList.toggle('is-me', p.seat === mySeat);
            } else {
                nameEl.textContent = '等待中...';
                el.classList.remove('occupied', 'is-me');
            }
        });
        document.getElementById('levelA').textContent = data.team_levels['0'] || '3';
        document.getElementById('levelB').textContent = data.team_levels['1'] || '3';
        document.getElementById('onStage').textContent =
            data.on_stage_team === 0 ? 'A队' : 'B队';

        const btnStart = document.getElementById('btnStart');
        const btnAddAI = document.getElementById('btnAddAI');
        btnStart.disabled = data.player_count < 4;
        btnAddAI.disabled = data.player_count >= 4 || data.game_active;
    });

    ws.on('game_state', data => {
        AppLogger.info('App', '收到游戏状态', {
            phase: data.phase,
            current_player_seat: data.current_player_seat,
            is_free_play: data.is_free_play,
            last_play_seat: data.last_play_seat,
            history_size: (data.play_history || []).length
        });
        if (!$game.classList.contains('active')) {
            showScreen('game');
        }
        renderGameState(data);
    });

    ws.on('game_action', data => {
        AppLogger.info('App', '收到游戏动作', data);
        if (data.action === 'game_started') {
            showScreen('game');
            document.getElementById('gameLog').innerHTML = '';
            addLog((data.restart ? '游戏重新开始！级牌: ' : '游戏开始！级牌: ') + data.level_rank);
            return;
        }
        const seat = data.seat;
        const player = gameState
            ? gameState.players.find(p => p.seat === seat)
            : null;
        const pn = player ? player.name : '玩家' + seat;

        if (data.action === 'cha') {
            addLog(pn + ' 叉牌！', 'log-cha');
        } else if (data.action === 'dian') {
            addLog(pn + ' 点牌！', 'log-dian');
        } else if (data.action === 'pass') {
            addLog(pn + ' 不出');
        } else if (data.action === 'cha_pass') {
            addLog(pn + ' 不叉');
        } else if (data.action === 'dian_pass') {
            addLog(pn + ' 不点');
        } else if (data.action === 'cha_all_pass') {
            addLog('没有人叉牌');
        } else if (data.action === 'dian_all_pass') {
            addLog('没有人点牌，叉牌成功');
        } else if (data.hand_type) {
            const hn = HAND_NAMES[data.hand_type.category] || '';
            const bombCls = ['BOMB3','BOMB4','JOKER_BOMB','SI_YAO_SI']
                .includes(data.hand_type.category) ? 'log-bomb' : '';
            addLog(pn + ' 出牌: ' + hn, bombCls);
        }

        if (data.player_finished) {
            addLog(pn + ' 出完牌了！');
        }
    });

    ws.on('round_end', data => {
        AppLogger.info('App', '收到结算', data);
        showResult(data);
    });

    function loadCheckpoints() {
        AppLogger.info('App', '加载checkpoint列表');
        fetch('/api/checkpoints')
            .then(r => r.json())
            .then(data => {
                const select = document.getElementById('aiCheckpointSelect');
                select.innerHTML = '';
                const files = data.checkpoints && data.checkpoints.length
                    ? data.checkpoints : ['default_rl.pt'];
                AppLogger.info('App', 'checkpoint列表加载完成', { checkpoints: files });
                files.forEach(name => {
                    const opt = document.createElement('option');
                    opt.value = name;
                    opt.textContent = name;
                    select.appendChild(opt);
                });
            })
            .catch(err => {
                AppLogger.warn('App', 'checkpoint列表加载失败', err);
            });
    }

    // 添加AI
    document.getElementById('btnAddAI').addEventListener('click', () => {
        const modelType = document.getElementById('aiModelSelect').value;
        const checkpoint = document.getElementById('aiCheckpointSelect').value;
        const model = modelType === 'rule' ? 'rule' : checkpoint;
        AppLogger.info('App', '点击添加AI', { model_type: modelType, checkpoint: checkpoint, model: model });
        document.getElementById('btnAddAI').disabled = true;
        ws.send('add_ai', { model: model });
    });

    // 开始游戏
    document.getElementById('btnStart').addEventListener('click', () => {
        AppLogger.info('App', '点击开始游戏', { room_id: currentRoomId });
        ws.send('start_game', {});
    });

    // 重新开始当前房间游戏，用于调试重新发牌
    document.getElementById('btnRestart').addEventListener('click', () => {
        AppLogger.info('App', '点击重新开始', { room_id: currentRoomId });
        document.getElementById('resultDialog').classList.add('hidden');
        ws.send('restart_game', {});
    });

    // 出牌
    document.getElementById('btnPlay').addEventListener('click', () => {
        if (selectedIndices.size === 0) return;
        AppLogger.info('App', '点击出牌', { card_indices: Array.from(selectedIndices) });
        ws.send('play_cards', {
            card_indices: Array.from(selectedIndices)
        });
    });

    // 不出
    document.getElementById('btnPass').addEventListener('click', () => {
        AppLogger.info('App', '点击不出');
        ws.send('pass_turn', {});
    });

    // 叉牌
    document.getElementById('btnChaYes').addEventListener('click', () => {
        ws.send('respond_cha', { do_cha: true });
        document.getElementById('chaDialog').classList.add('hidden');
    });
    document.getElementById('btnChaNo').addEventListener('click', () => {
        ws.send('respond_cha', { do_cha: false });
        document.getElementById('chaDialog').classList.add('hidden');
    });

    // 点牌
    document.getElementById('btnDianYes').addEventListener('click', () => {
        ws.send('respond_dian', { do_dian: true });
        document.getElementById('dianDialog').classList.add('hidden');
    });
    document.getElementById('btnDianNo').addEventListener('click', () => {
        ws.send('respond_dian', { do_dian: false });
        document.getElementById('dianDialog').classList.add('hidden');
    });

    // 下一局
    document.getElementById('btnNextRound').addEventListener('click', () => {
        document.getElementById('resultDialog').classList.add('hidden');
        ws.send('start_game', {});
    });
})();
