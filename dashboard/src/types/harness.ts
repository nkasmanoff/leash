export type TurnSummary = {
  req_id: string;
  session_id: string;
  created_at: number;
  finished_at: number;
  duration_s: number;
  model: string;
  clamp: boolean;
  thinking: boolean;
  fake_tools?: boolean;
  error: string | null;
  stats: {
    min: number;
    max: number;
    mean: number;
    n_tokens: number;
  };
  preview: {
    system_chars: number;
    last_user: string;
    input_tool_calls: number;
    tool_results: number;
    response_chars: number;
  };
  legacy_path?: string;
};

export type TraceToken = {
  i: number;
  token: string;
  token_id?: number;
  projection: number;
  capped?: boolean;
  ts: number;
  req_id?: string;
};

export type ConversationEntry = {
  index: number;
  role: string;
  kind: string;
  content: string;
  tool_calls?: Array<{
    id?: string;
    function?: { name?: string; arguments?: string };
  }>;
  tool_call_id?: string;
  name?: string;
};

export type ResponseSegment = {
  kind: "thinking" | "content" | string;
  start: number;
  end: number;
  text: string;
};

export type TurnDetail = {
  req_id: string;
  session_id: string;
  created_at: number;
  finished_at: number;
  duration_s: number;
  model: string;
  clamp: boolean;
  thinking: boolean;
  fake_tools?: boolean;
  stream?: boolean;
  error: string | null;
  legacy?: boolean;
  request: {
    temperature?: number;
    max_tokens?: number;
    tools?: unknown[];
    tool_choice?: unknown;
    messages: unknown[];
  };
  parsed: {
    system_prompt: string;
    conversation: ConversationEntry[];
    input_tool_calls: Array<{
      message_index: number;
      id?: string;
      name?: string;
      arguments?: string;
    }>;
    tool_results: Array<{
      message_index: number;
      tool_call_id?: string;
      name?: string;
      content: string;
    }>;
    response: {
      segments: ResponseSegment[];
      tool_markers: Array<{ kind: string; command?: string; at: number }>;
      text: string;
    };
  };
  response_text: string;
  stats: TurnSummary["stats"];
  tokens: TraceToken[];
};

export type SessionBundle = {
  session_id: string;
  turns: TurnDetail[];
};

export type TraceListResponse = {
  turns: TurnSummary[];
  sessions: Array<{
    session_id: string;
    req_ids: string[];
    turn_count: number;
  }>;
};
