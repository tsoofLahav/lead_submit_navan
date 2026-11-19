"use strict";

const APIClient = (() => {
  class APIError extends Error {
    constructor(detail) {
      super(detail?.message || "The request could not be completed.");
      this.code = detail?.code;
      this.retry = detail?.retry;
    }
  }

  async function request(path, data, onProgress) {
    const options = { credentials: "same-origin" };
    if (data !== undefined) {
      options.method = "POST";
      options.headers = { "Content-Type": "application/json", "X-Navan-Request": "1" };
      options.body = JSON.stringify(data);
    }
    let response;
    try {
      response = await fetch(path, options);
    } catch {
      throw new APIError({ code: "connection_lost", message: "Connection lost. Refresh the conversation to check what was saved." });
    }
    if (!response.ok) {
      const body = await response.json().catch(() => null);
      throw new APIError(body?.error);
    }
    if (!response.headers.get("Content-Type")?.startsWith("text/event-stream")) return response.json();

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    try {
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let boundary;
        while ((boundary = buffer.indexOf("\n\n")) !== -1) {
          const frame = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          if (frame.startsWith(":")) continue;
          const lines = frame.split("\n");
          const event = lines.find(line => line.startsWith("event: "))?.slice(7);
          const dataLine = lines.find(line => line.startsWith("data: "))?.slice(6);
          if (!event || !dataLine) continue;
          const payload = JSON.parse(dataLine);
          if (event === "result") return payload;
          if (event === "error") throw new APIError(payload);
          if (event === "progress") onProgress?.(payload);
        }
      }
    } catch (error) {
      if (error instanceof APIError) throw error;
    } finally {
      reader.cancel().catch(() => {});
      reader.releaseLock();
    }
    throw new APIError({ code: "connection_lost", message: "The stream ended early. Refresh the conversation to check what was saved." });
  }

  const thread = id => "/api/conversations/" + encodeURIComponent(id);
  return {
    async createConversation() { return (await request("/api/conversations", {})).conversation; },
    async listConversations() { return (await request("/api/conversations")).conversations; },
    getConversation(id, beforeId) {
      return request(thread(id) + (beforeId ? "?before_id=" + encodeURIComponent(beforeId) : ""));
    },
    async listCompanies() { return (await request("/api/companies")).companies; },
    chat(id, message, onProgress) { return request(thread(id) + "/chat", { message }, onProgress); },
    submit(id, type, data, onProgress) {
      return request(thread(id) + (type === "lead" ? "/leads" : "/customer-activity"), data, onProgress);
    },
    decide(id, evaluationId, accept) {
      return request(thread(id) + "/evaluations/" + encodeURIComponent(evaluationId) + "/decision", { accept });
    },
    retry(id, retry, onProgress) { return request(thread(id) + "/retry", retry, onProgress); },
  };
})();
