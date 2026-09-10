// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

/**
 * Quick single-model chat popup for the Source Models tab ("Play" button).
 * Opens a streaming conversation with one source. Deliberately feedback-free:
 * structured ratings live in the A/B tests and auto-eval flows.
 */

import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Textarea } from "@/components/ui/textarea";
import { authFetch } from "@/features/auth";
import { useT } from "@/i18n";
import { SentIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import {
  type BattlegroundSource,
  battlegroundApi,
} from "./api/battleground-api";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export function PlayChatDialog({
  source,
  open,
  onOpenChange,
}: {
  source: BattlegroundSource | null;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const t = useT();
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  // Reasoning models stream a thinking phase before any visible text; show it
  // instead of a silent bubble while that runs.
  const [thinking, setThinking] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  const startSession = useCallback(async (id: string) => {
    try {
      const session = await battlegroundApi.createPlaySession(id);
      setSessionId(session.session_id);
      setMessages([]);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
  }, []);

  useEffect(() => {
    if (open && source) {
      setMessages([]);
      setSessionId(null);
      void startSession(source.id);
    }
  }, [open, source, startSession]);

  useEffect(() => {
    const node = scrollRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [messages]);

  const send = async () => {
    const message = input.trim();
    if (!message || !sessionId || streaming) return;
    setInput("");
    setStreaming(true);
    setThinking(false);
    const history = [...messages];
    setMessages((prev) => [
      ...prev,
      { role: "user", content: message },
      { role: "assistant", content: "" },
    ]);
    try {
      const payloadMessages = [
        ...history,
        { role: "user", content: message },
      ].map((m) => ({ role: m.role, content: m.content }));
      const response = await authFetch(
        `/api/battleground/play/sessions/${sessionId}/chat`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ messages: payloadMessages, stream: true }),
        },
      );
      if (!response.ok || !response.body) {
        const body = await response.json().catch(() => null);
        throw new Error(
          (body && typeof body.detail === "string" && body.detail) ||
            `Chat failed (${response.status})`,
        );
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      const assistantIndex = history.length + 1;
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let idx: number;
        while ((idx = buffer.indexOf("\n\n")) >= 0) {
          const frame = buffer.slice(0, idx);
          buffer = buffer.slice(idx + 2);
          if (!frame.startsWith("data:")) continue;
          let event: Record<string, unknown>;
          try {
            event = JSON.parse(frame.slice(5).trim()) as Record<
              string,
              unknown
            >;
          } catch {
            continue;
          }
          const type = event.type as string;
          if (type === "delta" && typeof event.reasoning === "string") {
            setThinking(true);
          } else if (type === "delta" && typeof event.text === "string") {
            setThinking(false);
            const text = event.text;
            setMessages((prev) => {
              const next = [...prev];
              next[assistantIndex] = {
                ...next[assistantIndex],
                content: next[assistantIndex].content + text,
              };
              return next;
            });
          } else if (type === "done") {
            setMessages((prev) => {
              const next = [...prev];
              next[assistantIndex] = {
                role: "assistant",
                content: (event.content as string) ?? "",
              };
              return next;
            });
          } else if (type === "error") {
            setMessages((prev) => {
              const next = [...prev];
              next[assistantIndex] = {
                role: "assistant",
                content: `⚠ ${event.message}`,
              };
              return next;
            });
          }
        }
      }
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
      setMessages((prev) => prev.slice(0, -1));
    } finally {
      setStreaming(false);
      setThinking(false);
    }
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="flex h-[80vh] flex-col sm:max-w-2xl">
        <DialogHeader>
          <DialogTitle>
            {source
              ? t("battleground.playDialogTitle", { name: source.name })
              : t("battleground.selectSource")}
          </DialogTitle>
        </DialogHeader>

        <div
          ref={scrollRef}
          className="bg-muted/30 flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto rounded-xl border p-4"
        >
          {messages.length === 0 && (
            <p className="text-muted-foreground m-auto text-sm">
              {sessionId
                ? t("battleground.playHint")
                : t("battleground.connecting")}
            </p>
          )}
          {messages.map((message, index) =>
            message.role === "user" ? (
              <div
                key={index}
                className="bg-muted ml-auto max-w-[80%] rounded-2xl px-3.5 py-2 text-sm whitespace-pre-wrap"
              >
                {message.content}
              </div>
            ) : (
              <div key={index} className="max-w-[95%]">
                <div className="rounded-2xl border px-3.5 py-2 text-sm whitespace-pre-wrap">
                  {message.content}
                  {streaming &&
                    index === messages.length - 1 &&
                    !message.content &&
                    thinking && (
                      <span className="text-muted-foreground">
                        {t("battleground.thinking")}
                      </span>
                    )}
                  {streaming && index === messages.length - 1 && (
                    <span className="bg-primary ml-0.5 inline-block h-3.5 w-1.5 animate-pulse rounded-sm align-text-bottom" />
                  )}
                </div>
              </div>
            ),
          )}
        </div>

        <div className="flex items-end gap-2">
          <Textarea
            value={input}
            rows={1}
            placeholder={t("battleground.typeMessage")}
            disabled={!sessionId || streaming}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                void send();
              }
            }}
            className="max-h-40 min-h-9 resize-none"
          />
          <Button
            size="icon"
            disabled={!sessionId || streaming || !input.trim()}
            onClick={() => void send()}
          >
            <HugeiconsIcon icon={SentIcon} className="size-4" />
          </Button>
        </div>
      </DialogContent>
    </Dialog>
  );
}
