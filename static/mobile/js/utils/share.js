const ShareHelper = {
    getShareUrl(roomId) {
        const base = window.location.origin + '/mobile';
        return base + '?room=' + encodeURIComponent(roomId);
    },

    share(roomId) {
        const url = this.getShareUrl(roomId);

        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(url).then(function () {
                AppLogger.info('Share', 'Clipboard API复制成功');
            }).catch(function (err) {
                AppLogger.warn('Share', 'Clipboard API复制失败', err);
            });
            return true;
        }

        if (this._copySync(url)) {
            AppLogger.info('Share', 'execCommand复制成功');
            return true;
        }

        if (navigator.share) {
            navigator.share({
                title: '来四幺四牌桌一起玩！',
                text: '房间号：' + roomId,
                url: url
            }).catch(function () {});
            return true;
        }

        AppLogger.warn('Share', '所有分享方式均失败');
        return false;
    },

    _copySync(text) {
        var el = document.createElement('span');
        el.contentEditable = true;
        el.textContent = text;
        el.style.position = 'fixed';
        el.style.left = '-9999px';
        el.style.top = '0';
        el.style.webkitUserSelect = 'text';
        el.style.userSelect = 'text';
        document.body.appendChild(el);

        var range = document.createRange();
        range.selectNodeContents(el);
        var sel = window.getSelection();
        sel.removeAllRanges();
        sel.addRange(range);

        var success = false;
        try {
            success = document.execCommand('copy');
        } catch (e) {
            AppLogger.warn('Share', 'execCommand异常', e);
        }

        sel.removeAllRanges();
        document.body.removeChild(el);
        return success;
    }
};