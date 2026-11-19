// gta6-trophy-rarity-forecast-01 entry (Node 22, standard library only).
//
// Model: dilution-aware trophy rarity forecast.
//
// A trophy's Exophase earn rate is earners / players_tracked, and both move:
// players_tracked jumps when Exophase refreshes the page (launch night: 638
// players, every rate frozen for 4+ hours, then one refresh took it to 3,868
// players and every rate but the 100% one fell), while earners grow as players
// work through the story. So the program models the two counts separately
// instead of extrapolating the rate itself:
//
//   log(players_tracked) and log(earners_i), each fit by least squares
//   against u = ln(1 + hours since the 2026-11-19T00:00:00Z release) over the
//   last 24 hours of reads, then read at the target and divided back into a
//   rate. The fitted move is damped toward the last rate (regimes change over
//   a multi-day horizon), and the spread combines the fits' residuals, a
//   random-walk term in sqrt(days), and one refresh worth of jump surprise.
//
// When the whole history is one frozen page (no refresh ever seen, as in the
// practice cases), there is no trend to fit, so a documented refresh prior
// takes over: the page has been frozen for S hours, Exophase refreshes on a
// roughly 5-hour cadence, so a refresh before the target is likely, and at a
// refresh the rate falls by a factor that depends on how many of the newly
// tracked players could already hold the trophy (near zero for the platinum,
// 100% completion and the long grinds; a few percent for easy one-off
// trophies). The forecast is the two-component mixture (no refresh / refresh)
// and its quantiles are inverted from that mixture, so the band is honestly
// wide. Nothing here invents precision the data does not support: two days
// of launch data cannot pin a rate a week out, so the quantiles stay broad.
//
// Deterministic: no clock, no randomness, no network at runtime. Reads every
// *.in in the input directory (MAKEFUN_INPUT_DIR, else /input) and writes one
// *.out per case to MAKEFUN_OUTPUT_DIR (else /output). Never removes the
// output directory: only mkdirSync(out, { recursive: true }).

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const INPUT_DIR = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUTPUT_DIR = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

/** The quantile levels the contest scores, in the order each row lists them. */
const LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95];
/** The standard normal's quantile at each level. */
const Z = [-1.645, -0.674, 0, 0.674, 1.645];

/** Release moment: the trend clock u = ln(1 + hours since release) starts here. */
const RELEASE_MS = Date.parse('2026-11-19T00:00:00Z');
/** How long the last 24 hours of reads is, in milliseconds, for the trend fit. */
const WINDOW_MS = 24 * 3600 * 1000;
/** Exophase's refresh cadence, in hours: the page sat frozen 4h12m on launch night. */
const REFRESH_PERIOD_H = 5;
/** Random-walk spread, in logit units per square-root day (house baseline's). */
const SIGMA_RW = 0.5;
/** One refresh worth of jump surprise, in logit units, inside the spread. */
const SIGMA_JUMP = 0.35;
/** Floor / cap for the fitted spread, in logit units. */
const SPREAD_MIN = 0.35;
const SPREAD_MAX = 2.0;
/** How much of the fitted move survives damping, by reads in the fit window. */
const DAMP_MANY = 0.7;
const DAMP_FEW = 0.5;
/** A rate of exactly 100% (the opening mission) is forecast to stay there. */
const FULL_RATE = 99.99;

interface Trophy {
  id: number;
  name: string;
  description: string;
  tier: string;
}

interface Read {
  at: string;
  players_tracked: number;
  rates: Array<number | null>;
}

interface Case {
  id: string;
  at: string;
  horizon_minutes: number;
  target: string;
  levels: number[];
  trophies: Trophy[];
  history: Read[];
}

const clampPct = (x: number): number => Math.min(99.99, Math.max(0.01, x));
const logit = (pct: number): number => {
  const p = clampPct(pct) / 100;
  return Math.log(p / (1 - p));
};
const invLogit = (l: number): number => 100 / (1 + Math.exp(-l));
/** x rounded to 4 decimals. */
const round4 = (x: number): number => Math.round(x * 10_000) / 10_000;
/** Hours from the release to an ISO timestamp (0 when the timestamp is unusable). */
const hoursSinceRelease = (iso: string): number => {
  const ms = Date.parse(iso) - RELEASE_MS;
  return Number.isFinite(ms) ? ms / 3600_000 : 0;
};
/** The trend clock: log time since release, so early growth is steep and late growth flattens. */
const trendClock = (iso: string): number => Math.log(1 + Math.max(0, hoursSinceRelease(iso)));

const mean = (xs: number[]): number => xs.reduce((a, b) => a + b, 0) / xs.length;

/** Standard normal CDF (Abramowitz & Stegun 7.1.26); deterministic, no tables. */
const normCdf = (x: number): number => {
  const t = 1 / (1 + 0.2316419 * Math.abs(x));
  const d = 0.3989423 * Math.exp(-(x * x) / 2);
  const poly = t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))));
  const p = d * poly;
  return x > 0 ? 1 - p : p;
};

interface Fit {
  a: number;
  b: number;
  resid: number;
  n: number;
}

/** Least-squares line y = a + b*x, plus the residual standard deviation. */
const ols = (xs: number[], ys: number[]): Fit => {
  const n = xs.length;
  const mx = mean(xs);
  const my = mean(ys);
  let sxx = 0;
  let sxy = 0;
  for (let j = 0; j < n; j += 1) {
    sxx += (xs[j] - mx) * (xs[j] - mx);
    sxy += (xs[j] - mx) * (ys[j] - my);
  }
  const b = sxx > 1e-12 ? sxy / sxx : 0;
  const a = my - b * mx;
  let ss = 0;
  for (let j = 0; j < n; j += 1) {
    const e = ys[j] - (a + b * xs[j]);
    ss += e * e;
  }
  const resid = n > 2 ? Math.sqrt(ss / (n - 2)) : 0;
  return { a, b, resid, n };
};
/**
 * Relative weight of a trophy when the few new earners a refresh brings are
 * shared out: the story takes 60-80 hours and the platinum about 200, so in
 * launch week almost no newly tracked player holds the platinum, 100%
 * completion or the long grinds (250 stunts, gold in all 42 shooting ranges,
 * 100 WAiNK spots, 3 card sets, 50 challenges, 8 stranger strands, meditating
 * everywhere, exploring all of Leonida, buying every garage); story chapters
 * and strand finales are reachable only after hours of play; easy one-offs
 * (an arcade game, a cocktail, a bowling frame) are the most likely to
 * already be held. These are only relative weights — the absolute count comes
 * from the refresh totals below, so the forecast stays consistent with them.
 */
const refreshWeight = (t: Trophy): number => {
  const text = `${t.name} ${t.description}`.toLowerCase();
  if (t.tier === 'platinum') return 0;
  if (text.includes('100% completion')) return 0;
  if (/250 stunts|gold medal in all|100 locations|3 collectible card sets|50 challenges|8 mission strands|meditate at every|explore all of leonida|purchase all garage/.test(text)) return 0.0005;
  if (/complete chapter|complete the final mission|final mission for|complete breakfast run|complete the cartel heist|complete child's play|complete headliners|complete vcia takedown|complete operation savage glory/.test(text)) return 0.005;
  if (/win 20|discover 100 vehicles|rob 5 bank vaults|earn \$250,000|spend \$100,000|register 10 vehicles|win a top fun drone race/.test(text)) return 0.012;
  return 0.025;
};

/** Documented first refresh: players 638 -> 3,868, trophies earned 943 -> 4,276. */
const FIRST_REFRESH_N_OLD = 638;
const FIRST_REFRESH_N_NEW = 3868;
const FIRST_REFRESH_EARNED_OLD = 943;
const FIRST_REFRESH_EARNED_NEW = 4276;

/**
 * Quantiles of a two-component normal mixture on the logit scale, by
 * bisection on its CDF: weight w0 on N(mu0, s0), the rest on N(mu1, s1).
 * Deterministic; the CDF is monotone so bisection converges in fixed steps.
 */
const mixtureQuantiles = (
  mu0: number,
  s0: number,
  w0: number,
  mu1: number,
  s1: number,
): number[] => {
  const cdf = (x: number): number =>
    w0 * normCdf((x - mu0) / s0) + (1 - w0) * normCdf((x - mu1) / s1);
  return LEVELS.map((tau) => {
    let lo = -14;
    let hi = 14;
    for (let k = 0; k < 80; k += 1) {
      const mid = (lo + hi) / 2;
      if (cdf(mid) < tau) lo = mid;
      else hi = mid;
    }
    return (lo + hi) / 2;
  });
};

/**
 * The flat-history forecast: every read in the case is the same frozen page,
 * so no trend exists to fit. A refresh is pending (the page has been frozen
 * for `staleH` hours and refreshes roughly every REFRESH_PERIOD_H), and at
 * that refresh the rate falls by `factor`, the total-consistent dilution
 * factor for this trophy (see forecastCase: the documented refresh totals
 * leave only ~103 new earners across the 54 non-100% trophies, so every rate
 * falls to roughly a fifth of its frozen value). Extra refreshes inside a
 * longer horizon add the same factor again, at most twice more. The band is
 * the mixture's, so it covers both "no refresh yet" and "refreshed" outcomes.
 */
const flatForecast = (
  lastRate: number,
  staleH: number,
  horizonH: number,
  factor: number,
): number[] => {
  const yLast = logit(lastRate);
  const timeToNext = Math.max(0.5, REFRESH_PERIOD_H - staleH);
  const pRefresh = Math.min(1, horizonH / timeToNext);
  const extra = Math.min(2, Math.max(0, (horizonH - timeToNext) / REFRESH_PERIOD_H));
  const muRefreshed = yLast + Math.log(factor) * (1 + extra);
  const spreadRefreshed = Math.sqrt(0.55 * 0.55 + SIGMA_JUMP * SIGMA_JUMP);
  const logits = mixtureQuantiles(yLast, 0.12, 1 - pRefresh, muRefreshed, spreadRefreshed);
  return logits.map((l) => round4(invLogit(l)));
};
/** True when every read shows the same players and the same rates: a frozen page. */
const isFlat = (history: Read[]): boolean => {
  if (history.length < 2) return true;
  const first = JSON.stringify([history[0].players_tracked, history[0].rates]);
  return history.every((r) => JSON.stringify([r.players_tracked, r.rates]) === first);
};

/** A wide, honest fallback row when a trophy has no usable history at all. */
const FALLBACK_Q = [0.01, 0.05, 0.5, 5, 25];

/** Forecast one case: one row per trophy, five quantiles in percent. */
const forecastCase = (c: Case): Array<{ id: number; q: number[] }> => {
  if (c.history.length === 0) {
    return c.trophies.map((t) => ({ id: t.id, q: [...FALLBACK_Q] }));
  }
  const last = c.history[c.history.length - 1];
  const lastN = Math.max(1, last.players_tracked);
  const horizonH = c.horizon_minutes / 60;
  const horizonD = c.horizon_minutes / 1440;
  const uTarget = trendClock(c.target);
  const staleH = (Date.parse(c.at) - Date.parse(c.history[0].at)) / 3600_000;
  const flat = isFlat(c.history);

  // Total-consistent dilution factors for the flat case, one per trophy. The
  // documented first refresh took players 638 -> 3,868 and trophies earned
  // 943 -> 4,276; the 100% trophy alone accounts for all 3,230 new players,
  // so only 4,276 - 943 - 3,230 = 103 new earners are left for the other 54
  // trophies. Scale that budget to this case's counts, share it out by
  // difficulty weight, and turn each trophy's new earners into a rate factor.
  // (A trophy with no earners yet dilutes by the bare players ratio.)
  const lastRateOf = (i: number): number | null => {
    for (let j = c.history.length - 1; j >= 0; j -= 1) {
      const v = c.history[j].rates[i];
      if (v !== null && v !== undefined) return v;
    }
    return null;
  };
  const dilutionFactors: number[] = [];
  if (flat) {
    const nNew = lastN * (FIRST_REFRESH_N_NEW / FIRST_REFRESH_N_OLD);
    const earnedOld = c.trophies.reduce((s, _, i) => {
      const r = lastRateOf(i);
      return s + (r === null ? 0 : (r / 100) * lastN);
    }, 0);
    const earnedNew = earnedOld * (FIRST_REFRESH_EARNED_NEW / FIRST_REFRESH_EARNED_OLD);
    const budget = Math.max(0, earnedNew - earnedOld - (nNew - lastN));
    const weights = c.trophies.map((t, i) => {
      const r = lastRateOf(i);
      return r === null || r >= 99.99 ? 0 : refreshWeight(t);
    });
    const weightSum = weights.reduce((a, b) => a + b, 0);
    c.trophies.forEach((_, i) => {
      const r = lastRateOf(i);
      if (r === null || r >= 99.99) {
        dilutionFactors.push(1);
        return;
      }
      const eOld = (r / 100) * lastN;
      const alloc = weightSum > 0 ? (budget * weights[i]) / weightSum : 0;
      dilutionFactors.push(eOld > 0 ? ((eOld + alloc) / nNew) / (eOld / lastN) : lastN / nNew);
    });
  }

  // The trend fit uses the last 24 hours of reads (all of them when fewer),
  // so the current regime — dilution vs progression — is what extrapolates.
  const lastAt = Date.parse(last.at);
  const recent = c.history.filter((r) => Date.parse(r.at) >= lastAt - WINDOW_MS);
  const reads = recent.length >= 3 ? recent : c.history;
  const us = reads.map((r) => trendClock(r.at));
  const meanU = mean(us);

  // Players tracked: log-linear in u, slope clamped (growth only, never absurd).
  const logN = reads.map((r) => Math.log(Math.max(1, r.players_tracked)));
  const fitN = ols(us, logN);
  const bN = Math.min(2, Math.max(0, fitN.b));
  const logNT = mean(logN) - bN * meanU + bN * uTarget;

  return c.trophies.map((t, i) => {
    // The trophy's last observed rate, skipping reads that lacked it.
    const lastRate = lastRateOf(i);
    if (lastRate === null) return { id: t.id, q: [...FALLBACK_Q] };
    // The opening mission sits at 100% and every tracked player holds it.
    if (lastRate >= 99.99) return { id: t.id, q: [99.99, 100, 100, 100, 100] };
    // A frozen page carries no trend: the documented refresh prior decides.
    if (flat) {
      return { id: t.id, q: flatForecast(lastRate, staleH, horizonH, dilutionFactors[i]) };
    }

    // Earners per read (half a player floors a zero so the log stays defined),
    // fit in log space against u; earners never fall, so the slope is floored at 0.
    const xs: number[] = [];
    const ys: number[] = [];
    reads.forEach((r, j) => {
      const rate = r.rates[i];
      if (rate === null || rate === undefined) return;
      const earners = Math.max(0.25, (Math.max(0, rate) / 100) * Math.max(1, r.players_tracked));
      xs.push(us[j]);
      ys.push(Math.log(earners));
    });
    if (xs.length < 2) {
      const yLast = logit(lastRate);
      const s = Math.sqrt(SPREAD_MIN * SPREAD_MIN + SIGMA_JUMP * SIGMA_JUMP);
      return { id: t.id, q: Z.map((z) => round4(invLogit(yLast + z * s))) };
    }
    const fitE = ols(xs, ys);
    const bE = Math.min(2, Math.max(0, fitE.b));
    const meanX = mean(xs);
    const logET = mean(ys) - bE * meanX + bE * uTarget;
    const mFit = logit(Math.exp(logET - logNT) * 100);
    const yLast = logit(lastRate);
    // Damp the fitted move toward the last rate: the further the target and
    // the thinner the fit, the less a two-day trend should be trusted.
    const damp = xs.length >= 5 ? DAMP_MANY : DAMP_FEW;
    const m = yLast + damp * (mFit - yLast);
    // Spread: the fits' own residuals, the random walk over the horizon, and
    // one refresh of jump surprise — floored so a flat-looking fit never
    // claims false precision, capped so it never explodes.
    const sRaw = Math.sqrt(
      fitE.resid * fitE.resid +
        fitN.resid * fitN.resid +
        SIGMA_RW * SIGMA_RW * horizonD +
        SIGMA_JUMP * SIGMA_JUMP,
    );
    const s = Math.min(SPREAD_MAX, Math.max(SPREAD_MIN, sRaw));
    return { id: t.id, q: Z.map((z) => round4(invLogit(m + z * s))) };
  });
};

mkdirSync(OUTPUT_DIR, { recursive: true });
// Every case file, in name order, so two runs write the same bytes.
const inputs = readdirSync(INPUT_DIR).filter((f) => f.endsWith('.in')).sort();
for (const file of inputs) {
  const id = file.slice(0, -'.in'.length);
  const c = JSON.parse(readFileSync(join(INPUT_DIR, file), 'utf8')) as Case;
  if (c.levels.length !== LEVELS.length || c.levels.some((t, j) => t !== LEVELS[j])) throw new Error(`${file} asks for levels ${JSON.stringify(c.levels)}, not ${JSON.stringify(LEVELS)}`);
  if (c.history.length === 0) throw new Error(`${file} has no history`);
  const trophies = forecastCase(c);
  writeFileSync(join(OUTPUT_DIR, `${id}.out`), `${JSON.stringify({ trophies })}\n`);
}
