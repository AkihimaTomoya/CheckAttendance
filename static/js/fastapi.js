// static/js/fastapi-client.js

document.addEventListener('DOMContentLoaded', () => {
    // --- Lấy các phần tử HTML ---
    const video = document.getElementById('videoStream');
    const canvasOverlay = document.getElementById('canvasOverlay');
    const ctx = canvasOverlay.getContext('2d');
    const connectionStatusEl = document.getElementById('connectionStatus');
    const showBboxCheckbox = document.getElementById('showBbox');
    const showLabelCheckbox = document.getElementById('showLabel');
    const reloadFacebankBtn = document.getElementById('reloadFacebankBtn');
    const facebankStatus = document.getElementById('facebankStatus');
    const lastUpdateEl = document.getElementById('lastUpdate');
    const statusEl = document.getElementById('status');

    // --- Biến trạng thái ---
    let ws = null;
    let processingFrame = false;
    let faceLocations = [];  // Cập nhật liên tục từ server
    let recognitionData = {}; // Cập nhật định kỳ từ server
    let config = { show_bbox: true, show_label: true }; // Cấu hình mặc định

    // --- CÁC HÀM XỬ LÝ ---

    // 1. Kết nối WebSocket
    function connectWebSocket() {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${protocol}//${window.location.host}/ws`;
        ws = new WebSocket(wsUrl);

        ws.onopen = () => {
            connectionStatusEl.textContent = 'Connected';
            connectionStatusEl.style.color = '#4caf50';
        };

        ws.onclose = () => {
            connectionStatusEl.textContent = 'Disconnected. Retrying...';
            connectionStatusEl.style.color = '#f44336';
            setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            connectionStatusEl.textContent = 'Connection Error';
        };

        ws.onmessage = (event) => {
            const message = JSON.parse(event.data);
            handleWebSocketMessage(message);
        };
    }

    // 2. Xử lý tin nhắn từ WebSocket Server
    function handleWebSocketMessage(message) {
        switch (message.type) {
            case 'face_locations':
                faceLocations = message.data;
                break;
            case 'recognition_data':
                recognitionData = message.data;
                lastUpdateEl.textContent = `Recognition updated: ${new Date().toLocaleTimeString()}`;
                break;
            case 'config':
                config = message.data;
                showBboxCheckbox.checked = config.show_bbox;
                showLabelCheckbox.checked = config.show_label;
                break;
            default:
                console.log('Unknown message type:', message.type);
        }
    }

    // 3. Lấy stream từ camera và thiết lập canvas
    function setupCameraAndCanvas() {
        navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
            .then(stream => {
                video.srcObject = stream;
                video.play();
                video.addEventListener('loadedmetadata', () => {
                    // Đặt kích thước canvas bằng kích thước hiển thị của video
                    canvasOverlay.width = video.clientWidth;
                    canvasOverlay.height = video.clientHeight;
                    startAnimationLoop();
                    sendFramesToServer();
                });
            })
            .catch(err => {
                console.error("Camera access error:", err);
                alert("Could not access the camera. Please check permissions.");
            });
    }

    // 4. Vòng lặp vẽ lên canvas (Animation)
    function startAnimationLoop() {
        function animate() {
            drawOverlay();
            requestAnimationFrame(animate);
        }
        animate();
    }

    // 5. Hàm vẽ chính
    function drawOverlay() {
        // Cập nhật kích thước canvas nếu cửa sổ thay đổi
        if (canvasOverlay.width !== video.clientWidth || canvasOverlay.height !== video.clientHeight) {
            canvasOverlay.width = video.clientWidth;
            canvasOverlay.height = video.clientHeight;
        }

        ctx.clearRect(0, 0, canvasOverlay.width, canvasOverlay.height);

        statusEl.textContent = faceLocations.length > 0
            ? `Detected ${faceLocations.length} face(s)`
            : "No faces detected";

        // --- BẮT ĐẦU LOGIC SỬA LỖI BBOX ---
        // Tính toán tỉ lệ giữa kích thước video gốc và kích thước hiển thị
        const scaleX = video.clientWidth / video.videoWidth;
        const scaleY = video.clientHeight / video.videoHeight;
        // --- KẾT THÚC LOGIC SỬA LỖI BBOX ---

        faceLocations.forEach(face => {
            const [x1, y1, x2, y2] = face.bbox;
            const faceId = face.id;

            // Áp dụng tỉ lệ scale để có được tọa độ chính xác trên canvas
            const scaledX1 = x1 * scaleX;
            const scaledY1 = y1 * scaleY;
            const scaledWidth = (x2 - x1) * scaleX;
            const scaledHeight = (y2 - y1) * scaleY;

            const recognition = recognitionData[faceId] || { name: "Processing...", confidence: 0 };
            const { name, confidence } = recognition;

            const boxColor = name === 'Unknown' || name === 'Processing...' ? '#f44336' : '#4caf50';

            // Vẽ bounding box với tọa độ đã được scale
            if (config.show_bbox) {
                ctx.strokeStyle = boxColor;
                ctx.lineWidth = 2;
                ctx.strokeRect(scaledX1, scaledY1, scaledWidth, scaledHeight);
            }

            // Vẽ nhãn tên với tọa độ đã được scale
            if (config.show_label) {
                const displayText = name !== "Processing..."
                    ? `${name} (${(confidence * 100).toFixed(0)}%)`
                    : "Processing...";

                ctx.font = '16px Arial';
                const textWidth = ctx.measureText(displayText).width;

                ctx.fillStyle = boxColor;
                ctx.fillRect(scaledX1 - 1, scaledY1 - 22, textWidth + 10, 22);

                ctx.fillStyle = 'white';
                ctx.fillText(displayText, scaledX1 + 5, scaledY1 - 5);
            }
        });
    }

    // 6. Gửi frame video đến server
    function sendFramesToServer() {
        if (!ws || ws.readyState !== WebSocket.OPEN || processingFrame) {
            setTimeout(sendFramesToServer, 100); // Thử lại sau nếu chưa sẵn sàng
            return;
        }

        processingFrame = true;

        const captureCanvas = document.createElement('canvas');
        captureCanvas.width = video.videoWidth;
        captureCanvas.height = video.videoHeight;
        captureCanvas.getContext('2d').drawImage(video, 0, 0);

        const dataURL = captureCanvas.toDataURL('image/jpeg', 0.8); // Chất lượng 80%
        ws.send(dataURL);

        setTimeout(() => {
            processingFrame = false;
            sendFramesToServer(); // Lên lịch gửi frame tiếp theo
        }, 150); // Gửi khoảng 6-7 frame/giây
    }

    // 7. Cập nhật cấu hình UI
    async function updateConfig() {
        const newConfig = {
            show_bbox: showBboxCheckbox.checked,
            show_label: showLabelCheckbox.checked
        };
        await fetch('/config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(newConfig)
        });
        // Cập nhật config cục bộ ngay lập tức để UI phản hồi nhanh
        config = newConfig;
    }

    // 8. Cập nhật Facebank
    async function reloadFacebank() {
        facebankStatus.textContent = 'Updating...';
        facebankStatus.style.color = '#2196F3';
        const response = await fetch('/reload-facebank', { method: 'POST' });
        const result = await response.json();

        facebankStatus.textContent = result.message;
        facebankStatus.style.color = result.status === 'success' ? '#4caf50' : '#f44336';

        setTimeout(() => { facebankStatus.textContent = ''; }, 6000);
    }

    // --- GẮN EVENT LISTENERS ---
    showBboxCheckbox.addEventListener('change', updateConfig);
    showLabelCheckbox.addEventListener('change', updateConfig);
    reloadFacebankBtn.addEventListener('click', reloadFacebank);

    // --- KHỞI ĐỘNG ---
    setupCameraAndCanvas();
    connectWebSocket();
});