import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import "./index.css";

const App: React.FC = () => {
  const [query, setQuery] = useState("");
  const [response, setResponse] = useState("");
  const [history, setHistory] = useState<{ query: string; response: string }[]>(
    []
  );
  const [loading, setLoading] = useState(false);
  const socketRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await axios.get("http://localhost:8000/api/history");
        setHistory(res.data.history);
      } catch (error) {
        console.error("Error fetching history:", error);
      }
    };
    fetchHistory();
  }, []);

  useEffect(() => {
    const ws = new WebSocket("ws://localhost:8000/ws");
    ws.onopen = () => {
      console.log("WebSocket connection established");
    };
    ws.onmessage = (event) => {
      if (event.data === "History cleared") {
        setResponse("");
        setHistory([]);
        setLoading(false);
        return;
      }
      setResponse((prev) => prev + event.data);
      setLoading(false); // Stop loading once a message is received
    };
    ws.onerror = (error) => {
      console.error("WebSocket error:", error);
      setLoading(false);
    };
    ws.onclose = () => {
      console.log("WebSocket connection closed");
      setLoading(false);
    };
    socketRef.current = ws;
    return () => {
      ws.close();
    };
  }, []);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setResponse("");
    if (socketRef.current && socketRef.current.readyState === WebSocket.OPEN) {
      socketRef.current.send(query);
      setHistory((prevHistory) => [...prevHistory, { query, response: "" }]);
    } else {
      console.error("WebSocket is not open");
      setLoading(false);
    }
    setQuery("");
  };

  useEffect(() => {
    if (!loading && response) {
      setHistory((prevHistory) => {
        const updatedHistory = [...prevHistory];
        const lastIndex = updatedHistory.length - 1;
        if (lastIndex >= 0) {
          updatedHistory[lastIndex].response = response;
        }
        return updatedHistory;
      });
    }
  }, [response, loading]);

  return (
    <div className="App">
      <h1>Burger King GPT</h1>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Enter your query here"
        />
        <button type="submit" disabled={loading}>
          Ask
        </button>
      </form>
      {loading && <p>Loading...</p>}
      <div className="chat-history">
        {history.map((item, index) => (
          <div key={index} className="chat-item">
            <p>
              <strong>Query:</strong> {item.query}
            </p>
            <p>
              <strong>Response:</strong> {item.response}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
};

export default App;
