const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const cors = require("cors");
const axios = require("axios");
const FormData = require("form-data");
const multer = require("multer");

const app = express();
app.use(cors());
const upload = multer();
const server = http.createServer(app);
const io = new Server(server, { cors: { origin: "*" } });

// ---- CONNECT TO PYTHON API ---- //
const PYTHON_URL = "http://localhost:5001";

// ---- UPLOAD ENDPOINT ---- //
app.post("/analyze", upload.single("file"), async (req, res) => {
  try {
    const form = new FormData();
    form.append("file", req.file.buffer, {
      filename: req.file.originalname,
      contentType: req.file.mimetype,
    });

    // Send to Python YAMNet
    const response = await axios.post(`${PYTHON_URL}/predict`, form, {
      headers: form.getHeaders(),
    });

    const result = response.data;

    // Broadcast to all dashboard clients
    io.emit("sensor-data", {
      node: "Node-0" + (Math.floor(Math.random() * 8) + 1),
      type: result.threat_type,
      confidence: result.confidence,
      raw: result.raw_class,
      db: Math.floor(50 + Math.random() * 40),
      timestamp: new Date(),
    });

    res.json(result);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// ---- SOCKET CONNECTION ---- //
io.on("connection", (socket) => {
  console.log("Client connected");
  socket.on("disconnect", () => console.log("Client disconnected"));
});

app.get("/", (req, res) => res.send("Forest Guardian Backend Running"));

server.listen(3000, () =>
  console.log("Server running on http://localhost:3000"),
);
