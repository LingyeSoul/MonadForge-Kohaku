/**
 * The `npm ci` argv for a given npm major and offline flag.
 *
 * Kept in its own module so the npm-12 `allow-remote` gate can be tested as
 * behaviour instead of by grepping `ensure_npm_deps.mjs`.
 */
export function installArgs(major, offline) {
  const args = [
    "ci",
    "--ignore-scripts",
    offline ? "--offline" : "--prefer-offline",
    "--no-audit",
    "--no-fund",
  ];
  // npm >= 12 defaults `allow-remote=none`, which refuses URL tarballs whose
  // host differs from the configured registry (EALLOWREMOTE). The lockfile gate
  // in ensure_npm_deps.mjs already approved every URL and integrity hash, so
  // npm may fetch exactly those.
  if (major >= 12) args.push("--allow-remote=all");
  return args;
}
