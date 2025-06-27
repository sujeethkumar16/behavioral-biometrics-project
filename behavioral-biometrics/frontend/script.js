
let distances = [], delays = [], keys = [], lastX = 0, lastY = 0, lastClick = 0;

document.addEventListener('mousemove', (e) => {
    let dx = e.clientX - lastX, dy = e.clientY - lastY;
    distances.push(Math.sqrt(dx*dx + dy*dy));
    lastX = e.clientX; lastY = e.clientY;
});

document.addEventListener('click', () => {
    let now = Date.now();
    if (lastClick) delays.push((now - lastClick)/1000);
    lastClick = now;
});

document.getElementById("inputField").addEventListener('keydown', (e) => {
    keys.push(Date.now());
});

function getAvg(arr) {
    return arr.length ? arr.reduce((a, b) => a + b) / arr.length : 0;
}

function sendData() {
    const avg_speed = getAvg(distances);
    const avg_click_delay = getAvg(delays);
    const movement_variance = getAvg(distances.map(d => Math.pow(d - avg_speed, 2)));
    const key_intervals = keys.slice(1).map((t, i) => t - keys[i]);
    const avg_key_delay = getAvg(key_intervals);

    fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ avg_speed, avg_click_delay, movement_variance, avg_key_delay })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById('result').textContent = data.fraud ? "⚠️ Fraudulent" : "✅ Legitimate";
    });
}
