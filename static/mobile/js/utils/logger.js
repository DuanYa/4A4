const AppLogger = (function() {
    const LEVELS = { DEBUG: 10, INFO: 20, WARN: 30, ERROR: 40, OFF: 100 };
    let currentLevel = LEVELS.INFO;

    function shouldLog(level) {
        return LEVELS[level] >= currentLevel;
    }

    function log(level, scope, message, data) {
        if (!shouldLog(level)) return;
        const prefix = `[4A4-Mobile][${level}][${scope}] ${message}`;
        if (data !== undefined) {
            console.log(prefix, data);
        } else {
            console.log(prefix);
        }
    }

    return {
        setLevel(level) {
            const upper = String(level || '').toUpperCase();
            if (LEVELS[upper] !== undefined) {
                currentLevel = LEVELS[upper];
                console.log(`[4A4-Mobile][INFO][Logger] 日志级别已设置为 ${upper}`);
            }
        },
        debug(scope, message, data) { log('DEBUG', scope, message, data); },
        info(scope, message, data) { log('INFO', scope, message, data); },
        warn(scope, message, data) { log('WARN', scope, message, data); },
        error(scope, message, data) { log('ERROR', scope, message, data); },
    };
})();
