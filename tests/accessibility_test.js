const path = require('path');
const fs = require('fs');
const { execSync } = require('child_process');
const { chromium } = require('playwright');
const { AxeBuilder } = require('@axe-core/playwright');

(async () => {
  const target = process.env.A11Y_TARGET || 'build/embedded_ui.html';
  if (!target.startsWith('http') && !fs.existsSync(target)) {
    console.log(`A11Y target missing: ${target}. Generating embedded HTML...`);
    execSync('python scripts/build_embedded_html.py', { stdio: 'inherit' });
  }
  const url = target.startsWith('http') ? target : `file://${path.resolve(target)}`;

  const browser = await chromium.launch();
  const context = await browser.newContext();
  const page = await context.newPage();
  await page.goto(url);

  const results = await new AxeBuilder({ page }).analyze();
  console.log(`Accessibility violations: ${results.violations.length}`);

  if (results.violations.length > 0) {
    console.log(JSON.stringify(results.violations, null, 2));
    await context.close();
    await browser.close();
    process.exit(1);
  }

  await context.close();
  await browser.close();
})();
