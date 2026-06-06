/**
 * 前端统一日志器。
 * 默认级别为 INFO。需要关闭时在浏览器控制台执行：
 *   AppLogger.setLevel('OFF')
 * 需要更详细调试时执行：
 *   AppLogger.setLevel('DEBUG')
 */
(function() {
    const LEVELS = { DEBUG: 10, INFO: 20, WARN: 30, ERROR: 40, OFF: 100 };
    let currentLevel = LEVELS.INFO;

    function shouldLog(level) {
        return LEVELS[level] >= currentLevel;
    }

    function log(level, scope, message, data) {
        if (!shouldLog(level)) return;
        const prefix = `[4A4][${level}][${scope}] ${message}`;
        if (data !== undefined) {
            console.log(prefix, data);
        } else {
            console.log(prefix);
        }
    }

    window.AppLogger = {
        setLevel(level) {
            const upper = String(level || '').toUpperCase();
            if (LEVELS[upper] !== undefined) {
                currentLevel = LEVELS[upper];
                console.log(`[4A4][INFO][Logger] 日志级别已设置为 ${upper}`);
            }
        },
        debug(scope, message, data) { log('DEBUG', scope, message, data); },
        info(scope, message, data) { log('INFO', scope, message, data); },
        warn(scope, message, data) { log('WARN', scope, message, data); },
        error(scope, message, data) { log('ERROR', scope, message, data); },
    };
})();
