const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const cors = require("cors");

const app = express();
app.use(cors());

const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*" },
});

// ---- SENSOR SIMULATION ---- //
const threats = [
  { type: "chainsaw", conf: 90 },
  { type: "gunshot", conf: 85 },
  { type: "vehicle", conf: 75 },
  { type: "human", conf: 65 },
  { type: "ambient", conf: 100 },
];

function generateSensorData() {
  const t = threats[Math.floor(Math.random() * threats.length)];

  return {
    node: "Node-0" + (Math.floor(Math.random() * 8) + 1),
    type: t.type,
    confidence: t.conf,
    db: Math.floor(50 + Math.random() * 40),
    timestamp: new Date(),
  };
}

// ---- SOCKET CONNECTION ---- //
io.on("connection", (socket) => {
  console.log("Client connected");

  const interval = setInterval(() => {
    const data = generateSensorData();
    socket.emit("sensor-data", data);
  }, 2000);

  socket.on("disconnect", () => {
    clearInterval(interval);
    console.log("Client disconnected");
  });
});

// ---- API ---- //
app.get("/", (req, res) => {
  res.send("Forest Guardian Backend Running");
});

server.listen(3000, () => {
  console.log("Server running on http://localhost:3000");
});
