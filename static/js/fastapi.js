// static/js/fastapi.js

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
    const thresholdSlider = document.getElementById('thresholdSlider');
    const thresholdValue = document.getElementById('thresholdValue');
    const applyThresholdBtn = document.getElementById('applyThreshold');

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
            console.log('WebSocket connected successfully');
        };

        ws.onclose = () => {
            connectionStatusEl.textContent = 'Disconnected. Retrying...';
            connectionStatusEl.style.color = '#f44336';
            setTimeout(connectWebSocket, 3000);
        };

        ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            connectionStatusEl.textContent = 'Connection Error';
            connectionStatusEl.style.color = '#f44336';
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
        navigator.mediaDevices.getUserMedia({ 
            video: { 
                width: { ideal: 1280 }, 
                height: { ideal: 720 } 
            } 
        })
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
            statusEl.textContent = "Camera access denied";
            statusEl.style.color = '#f44336';
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

        if (faceLocations.length === 0) {
            statusEl.style.color = '#666';
        } else {
            statusEl.style.color = '#4c7faf';
        }

        // Tính toán tỉ lệ giữa kích thước video gốc và kích thước hiển thị
        const scaleX = video.clientWidth / video.videoWidth;
        const scaleY = video.clientHeight / video.videoHeight;

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

                // Vẽ nền cho text
                ctx.fillStyle = boxColor;
                ctx.fillRect(scaledX1 - 1, scaledY1 - 22, textWidth + 10, 22);

                // Vẽ text
                ctx.fillStyle = 'white';
                ctx.fillText(displayText, scaledX1 + 5, scaledY1 - 5);
            }
        });
    }

    // 6. Gửi frame video đến server
    function sendFramesToServer() {
        if (!ws || ws.readyState !== WebSocket.OPEN) {
            setTimeout(sendFramesToServer, 100);
            return;
        }

        if (processingFrame) {
            setTimeout(sendFramesToServer, 50);
            return;
        }

        processingFrame = true;

        try {
            const captureCanvas = document.createElement('canvas');
            captureCanvas.width = video.videoWidth;
            captureCanvas.height = video.videoHeight;
            const captureCtx = captureCanvas.getContext('2d');
            captureCtx.drawImage(video, 0, 0);

            const dataURL = captureCanvas.toDataURL('image/jpeg', 0.8);
            ws.send(dataURL);
        } catch (error) {
            console.error('Error sending frame:', error);
        }

        setTimeout(() => {
            processingFrame = false;
            sendFramesToServer();
        }, 150); // Gửi khoảng 6-7 frame/giây
    }

    // 7. Xử lý sự kiện checkbox hiển thị
    function setupEventHandlers() {
        // Xử lý checkbox show/hide bounding box
        showBboxCheckbox.addEventListener('change', async () => {
            const newConfig = {
                show_bbox: showBboxCheckbox.checked,
                show_label: showLabelCheckbox.checked
            };

            try {
                const response = await fetch('/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(newConfig)
                });

                if (response.ok) {
                    config = newConfig;
                    console.log('Config updated:', config);
                } else {
                    console.error('Failed to update config');
                }
            } catch (error) {
                console.error('Error updating config:', error);
            }
        });

        // Xử lý checkbox show/hide label
        showLabelCheckbox.addEventListener('change', async () => {
            const newConfig = {
                show_bbox: showBboxCheckbox.checked,
                show_label: showLabelCheckbox.checked
            };

            try {
                const response = await fetch('/config', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(newConfig)
                });

                if (response.ok) {
                    config = newConfig;
                    console.log('Config updated:', config);
                } else {
                    console.error('Failed to update config');
                }
            } catch (error) {
                console.error('Error updating config:', error);
            }
        });

        // Xử lý nút reload facebank
        reloadFacebankBtn.addEventListener('click', async () => {
            reloadFacebankBtn.disabled = true;
            reloadFacebankBtn.textContent = 'Reloading...';
            facebankStatus.textContent = 'Processing...';
            facebankStatus.style.color = '#ff9800';

            try {
                const response = await fetch('/reload-facebank', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' }
                });

                const result = await response.json();

                if (result.status === 'success') {
                    facebankStatus.textContent = result.message;
                    facebankStatus.style.color = '#4caf50';
                } else {
                    facebankStatus.textContent = `Error: ${result.message}`;
                    facebankStatus.style.color = '#f44336';
                }
            } catch (error) {
                console.error('Error reloading facebank:', error);
                facebankStatus.textContent = 'Network error occurred';
                facebankStatus.style.color = '#f44336';
            } finally {
                reloadFacebankBtn.disabled = false;
                reloadFacebankBtn.textContent = 'Reload Facebank';
            }
        });

        // Xử lý slider threshold
        thresholdSlider.addEventListener('input', () => {
            thresholdValue.textContent = thresholdSlider.value;
        });

        // Xử lý nút apply threshold
        applyThresholdBtn.addEventListener('click', async () => {
            const threshold = parseFloat(thresholdSlider.value);

            try {
                const response = await fetch('/threshold', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ threshold: threshold })
                });

                const result = await response.json();

                if (result.status === 'success') {
                    console.log(`Threshold updated from ${result.old_threshold} to ${result.new_threshold}`);
                    alert(`Threshold updated to ${result.new_threshold}`);
                } else {
                    console.error('Failed to update threshold');
                    alert('Failed to update threshold');
                }
            } catch (error) {
                console.error('Error updating threshold:', error);
                alert('Error updating threshold');
            }
        });
    }

    // 8. Tải threshold hiện tại từ server
    async function loadCurrentThreshold() {
        try {
            const response = await fetch('/threshold');
            const result = await response.json();

            if (result.threshold) {
                thresholdSlider.value = result.threshold;
                thresholdValue.textContent = result.threshold;
                console.log('Current threshold loaded:', result.threshold);
            }
        } catch (error) {
            console.error('Error loading current threshold:', error);
        }
    }

    // --- KHỞI TẠO ỨNG DỤNG ---
    function init() {
        console.log('Initializing Face Recognition App...');

        // Thiết lập camera và canvas
        setupCameraAndCanvas();

        // Thiết lập các event handlers
        setupEventHandlers();

        // Tải threshold hiện tại
        loadCurrentThreshold();

        // Kết nối WebSocket
        connectWebSocket();

        console.log('Face Recognition App initialized successfully!');
    }

    // Bắt đầu ứng dụng
    init();
});
