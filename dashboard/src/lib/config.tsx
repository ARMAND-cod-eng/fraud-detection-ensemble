import { createContext, useContext } from "react";
import type { PublicConfig } from "./api";

export const DEFAULT_CONFIG: PublicConfig = {
  version: "", environment: "local", demo_mode: false, max_replay_tps: 1000, max_replay_count: 200_000,
  policy_changes_allowed: true, demo_note: "", data_attribution: "",
  profile: "creditcard", currency: "EUR", scenarios: {}, linked_activity: false,
};

export const ConfigContext = createContext<PublicConfig>(DEFAULT_CONFIG);
export const usePublicConfig = () => useContext(ConfigContext);
