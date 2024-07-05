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
      console.log("Received message from WebSocket:", event.data);
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
    console.log("Sending query:", query);
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
        console.log("Updated history:", updatedHistory);
        return updatedHistory;
      });
    }
  }, [response, loading]);

  const parseResponse = (response: string) => {
    const parts = response.split("\n\nSources: ");
    const mainResponse = parts[0];
    const sources = parts[1] ? parts[1].split(", ") : [];
    return { mainResponse, sources };
  };

  return (
    <div className="App">
      <h1>XML local chroma gpt</h1>
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
        {history.map((item, index) => {
          const { mainResponse, sources } = parseResponse(item.response);
          return (
            <div key={index} className="chat-item">
              <p>
                <strong>Query:</strong> {item.query}
              </p>
              <p>
                <strong>Response:</strong> {mainResponse}
              </p>
              {sources.length > 0 && (
                <p>
                  <strong>Sources:</strong>
                  <ul>
                    {sources.map((source, i) => {
                      const match = source.match(/\[(.*)\]\((.*)\)/);
                      if (match) {
                        const [_, title, link] = match;
                        const fullLink = `http://localhost:8000${link}`;
                        return (
                          <li key={i}>
                            <a
                              href={fullLink}
                              target="_blank"
                              rel="noopener noreferrer"
                            >
                              {title}
                            </a>
                          </li>
                        );
                      } else {
                        return <li key={i}>{source}</li>;
                      }
                    })}
                  </ul>
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};

export default App;
