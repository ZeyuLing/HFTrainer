#!/usr/bin/env node
/* Headless Three.js recorder for the multi-representation demo.
 *
 * Drives motion_annot_web/repr_convert_demo /record (5-panel Three.js viewer)
 * with the bundled puppeteer chromium, captures one PNG per frame into
 * <out>/<case>/frame_XXXX.png. Stitch to MP4/GIF with ffmpeg afterwards.
 *
 * Usage:
 *   node scripts/demo/record_threejs.js --url http://127.0.0.1:8099 \
 *        --cases 000000,000019,000021 --out /tmp/threejs_frames --max 120
 */
const path = require('path');
const fs = require('fs');
const puppeteer = require('/root/.npm/_npx/07806fb36c73e358/node_modules/puppeteer');

function arg(name, def) {
  const i = process.argv.indexOf('--' + name);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}

const URL = arg('url', 'http://127.0.0.1:8099');
const CASES = arg('cases', '000000,000019,000021').split(',').map(s => s.trim()).filter(Boolean);
const OUT = arg('out', '/tmp/threejs_frames');
// default high so the native 30 fps mesh length is kept (capping distorts the
// real-time playback speed: a 6 s / 180-frame clip squeezed to 120 plays 1.5x fast)
const MAX = parseInt(arg('max', '600'));

const sleep = ms => new Promise(r => setTimeout(r, ms));

async function recordCase(browser, sid) {
  const page = await browser.newPage();
  page.on('console', m => { const t = m.text(); if (/error|Error/.test(t)) console.log('  [page]', t); });
  await page.setViewport({ width: 4000, height: 760, deviceScaleFactor: 1 });
  const url = `${URL}/record?case=${sid}&max=${MAX}`;
  await page.goto(url, { waitUntil: 'networkidle0', timeout: 120000 });
  // wait for boot
  await page.waitForFunction('window.READY === true || window.BOOT_ERROR', { timeout: 120000 });
  const err = await page.evaluate(() => window.BOOT_ERROR || null);
  if (err) { console.log('  BOOT_ERROR:', err); await page.close(); return; }
  const num = await page.evaluate(() => window.NUM_FRAMES);
  const strip = await page.$('#strip');
  const dir = path.join(OUT, sid);
  fs.mkdirSync(dir, { recursive: true });
  console.log(`  ${sid}: ${num} frames -> ${dir}`);
  for (let i = 0; i < num; i++) {
    await page.evaluate(t => window.renderFrame(t), i);
    await sleep(8);
    const f = path.join(dir, 'frame_' + String(i).padStart(4, '0') + '.png');
    await strip.screenshot({ path: f });
    if (i % 30 === 0) console.log(`    frame ${i}/${num}`);
  }
  await page.close();
  console.log(`  ${sid}: done`);
}

(async () => {
  const browser = await puppeteer.launch({
    executablePath: puppeteer.executablePath(),
    headless: true,
    args: [
      '--no-sandbox', '--disable-setuid-sandbox', '--disable-dev-shm-usage',
      '--use-gl=swiftshader', '--enable-webgl', '--ignore-gpu-blacklist',
      '--enable-unsafe-swiftshader',
    ],
  });
  for (const sid of CASES) {
    try { await recordCase(browser, sid); }
    catch (e) { console.log('  FAILED', sid, e.message); }
  }
  await browser.close();
})();
