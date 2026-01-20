import 'dotenv/config';
import express from 'express';

const PORT = parseInt(process.env.PORT || '8000', 10);

const app = express();
app.use(express.json());

app.get('/health', (_req, res) => {
  res.json({ status: 'ok', timestamp: new Date().toISOString() });
});

app.post('/mcp', (_req, res) => {
  res.json({ status: 'minimal server running' });
});

app.listen(PORT, () => {
  console.log(`Minimal server listening on port ${PORT}`);
});
