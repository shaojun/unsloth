// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import { SentIcon, ThumbsDownIcon, ThumbsUpIcon } from "@hugeicons/core-free-icons";
import { HugeiconsIcon } from "@hugeicons/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { authFetch } from "@/features/auth";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { useT } from "@/i18n";
import { cn } from "@/lib/utils";
import {
  FEEDBACK_TAGS,
  type PlaygroundSource,
  playgroundApi,
} from "./api/playground-api";

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  responseId?: string;
  feedback?: { rating: "good" | "bad"; tags: string[]; comment: string | null } | null;
}

export function PlayPanel() {
  const t = useT();
  const [sources, setSources] = useState<PlaygroundSource[]>([]);
  const [sourceId, setSourceId] = useState<string>("");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(false);
  // Reasoning models stream a thinking phase before any visible text; show it
  // instead of a silent bubble while that runs.
  const [thinking, setThinking] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    void playgroundApi
      .listSources()
      .then((list) => setSources(list))
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    const node = scrollRef.current;
    if (node) node.scrollTop = node.scrollHeight;
  }, [messages]);

  const startSession = useCallback(
    async (id: string) => {
      setSourceId(id);
      try {
        const session = await playgroundApi.createPlaySession(id);
        setSessionId(session.session_id);
        setMessages([]);
      } catch (error) {
        toast.error(error instanceof Error ? error.message : String(error));
      }
    },
    [],
  );

  const send = async () => {
    const message = input.trim();
    if (!message || !sessionId || streaming) return;
    setInput("");
    setStreaming(true);
    setThinking(false);
    const history = [...messages];
    setMessages((prev) => [...prev, { role: "user", content: message }, { role: "assistant", content: "" }]);
    try {
      const payloadMessages = [...history, { role: "user", content: message }].map(
        (m) => ({ role: m.role, content: m.content }),
      );
      const response = await authFetch(
        `/api/playground/play/sessions/${sessionId}/chat`,
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
            event = JSON.parse(frame.slice(5).trim()) as Record<string, unknown>;
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
              next[assistantIndex] = { ...next[assistantIndex], content: next[assistantIndex].content + text };
              return next;
            });
          } else if (type === "done") {
            setMessages((prev) => {
              const next = [...prev];
              next[assistantIndex] = {
                role: "assistant",
                content: (event.content as string) ?? "",
                responseId: event.response_id as string,
                feedback: null,
              };
              return next;
            });
          } else if (type === "error") {
            setMessages((prev) => {
              const next = [...prev];
              next[assistantIndex] = {
                role: "assistant",
                content: `⚠ ${event.message}`,
                feedback: null,
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

  const rate = async (
    index: number,
    rating: "good" | "bad",
    tags: string[],
    comment: string,
  ) => {
    const message = messages[index];
    if (!message?.responseId || !sessionId) return;
    try {
      await playgroundApi.submitFeedback({
        session_id: sessionId,
        kind: "rating",
        response_id: message.responseId,
        rating,
        tags,
        comment: comment || null,
      });
      setMessages((prev) => {
        const next = [...prev];
        next[index] = { ...next[index], feedback: { rating, tags, comment: comment || null } };
        return next;
      });
      toast.success(t("playground.feedbackSaved"));
    } catch (error) {
      toast.error(error instanceof Error ? error.message : String(error));
    }
  };

  return (
    <div className="mx-auto flex h-full w-full max-w-3xl flex-col gap-4 p-4 sm:p-6">
      <div className="flex items-center gap-2">
        <Select
          value={sourceId}
          onValueChange={(value) => void startSession(value)}
        >
          <SelectTrigger className="w-64">
            <SelectValue placeholder={t("playground.selectSource")} />
          </SelectTrigger>
          <SelectContent>
            {sources.map((source) => (
              <SelectItem key={source.id} value={source.id}>
                {source.name}
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
        {sessionId && (
          <Button
            variant="outline"
            size="sm"
            onClick={() => void startSession(sourceId)}
          >
            {t("playground.newChat")}
          </Button>
        )}
      </div>

      <div
        ref={scrollRef}
        className="bg-muted/30 flex min-h-0 flex-1 flex-col gap-3 overflow-y-auto rounded-xl border p-4"
      >
        {messages.length === 0 && (
          <p className="text-muted-foreground m-auto text-sm">
            {sessionId ? t("playground.playHint") : t("playground.selectSourceHint")}
          </p>
        )}
        {messages.map((message, index) =>
          message.role === "user" ? (
            <div key={index} className="bg-muted ml-auto max-w-[80%] rounded-2xl px-3.5 py-2 text-sm whitespace-pre-wrap">
              {message.content}
            </div>
          ) : (
            <div key={index} className="max-w-[95%]">
              <div className="rounded-2xl border px-3.5 py-2 text-sm whitespace-pre-wrap">
                {message.content}
                {streaming && index === messages.length - 1 && !message.content && thinking && (
                  <span className="text-muted-foreground">{t("playground.thinking")}</span>
                )}
                {streaming && index === messages.length - 1 && (
                  <span className="bg-primary ml-0.5 inline-block h-3.5 w-1.5 animate-pulse rounded-sm align-text-bottom" />
                )}
              </div>
              {message.responseId && !streaming && (
                <FeedbackControls
                  initial={message.feedback}
                  onSubmit={(rating, tags, comment) =>
                    void rate(index, rating, tags, comment)
                  }
                />
              )}
            </div>
          ),
        )}
      </div>

      <div className="flex items-end gap-2">
        <Textarea
          value={input}
          rows={1}
          placeholder={t("playground.typeMessage")}
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
        <Button size="icon" disabled={!sessionId || streaming || !input.trim()} onClick={() => void send()}>
          <HugeiconsIcon icon={SentIcon} className="size-4" />
        </Button>
      </div>
    </div>
  );
}

function FeedbackControls({
  initial,
  onSubmit,
}: {
  initial?: { rating: "good" | "bad"; tags: string[]; comment: string | null } | null;
  onSubmit: (rating: "good" | "bad", tags: string[], comment: string) => void;
}) {
  const t = useT();
  const [open, setOpen] = useState(false);
  const [rating, setRating] = useState<"good" | "bad" | null>(initial?.rating ?? null);
  const [tags, setTags] = useState<string[]>(initial?.tags ?? []);
  const [comment, setComment] = useState(initial?.comment ?? "");
  const [saved, setSaved] = useState(Boolean(initial));

  if (saved && rating) {
    return (
      <p className="text-muted-foreground mt-1 text-xs">
        {rating === "good" ? "👍" : "👎"} {t("playground.feedbackSaved")}
      </p>
    );
  }

  return (
    <div className="mt-1.5 flex flex-col gap-2">
      <div className="flex gap-1.5">
        <button
          type="button"
          className={cn(
            "rounded-full border px-2.5 py-1 text-xs transition-colors",
            rating === "good"
              ? "border-emerald-500 bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
              : "text-muted-foreground hover:border-foreground/30",
          )}
          onClick={() => {
            setRating("good");
            setOpen(true);
          }}
        >
          <HugeiconsIcon icon={ThumbsUpIcon} className="mr-1 size-3" />
          {t("playground.good")}
        </button>
        <button
          type="button"
          className={cn(
            "rounded-full border px-2.5 py-1 text-xs transition-colors",
            rating === "bad"
              ? "border-red-500 bg-red-500/10 text-red-600 dark:text-red-400"
              : "text-muted-foreground hover:border-foreground/30",
          )}
          onClick={() => {
            setRating("bad");
            setOpen(true);
          }}
        >
          <HugeiconsIcon icon={ThumbsDownIcon} className="mr-1 size-3" />
          {t("playground.bad")}
        </button>
      </div>
      {open && rating && (
        <div className="flex flex-col gap-2 rounded-lg border p-2">
          <div className="flex flex-wrap gap-1">
            {FEEDBACK_TAGS.map((tag) => (
              <button
                key={tag}
                type="button"
                className={cn(
                  "rounded-full border px-2 py-0.5 text-[11px] transition-colors",
                  tags.includes(tag)
                    ? "border-primary text-primary bg-primary/10"
                    : "text-muted-foreground hover:border-foreground/30",
                )}
                onClick={() =>
                  setTags((prev) =>
                    prev.includes(tag) ? prev.filter((item) => item !== tag) : [...prev, tag],
                  )
                }
              >
                {tag}
              </button>
            ))}
          </div>
          <Textarea
            value={comment}
            rows={2}
            placeholder={t("playground.comment")}
            onChange={(event) => setComment(event.target.value)}
            className="text-xs"
          />
          <Button
            size="sm"
            className="self-end"
            onClick={() => {
              onSubmit(rating, tags, comment);
              setSaved(true);
            }}
          >
            {t("playground.saveFeedback")}
          </Button>
        </div>
      )}
    </div>
  );
}
