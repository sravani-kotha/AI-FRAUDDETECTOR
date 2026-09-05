const form = document.querySelector('#score-form');
const result = document.querySelector('#result');
const error = document.querySelector('#error');
const health = document.querySelector('#health');

function valuesFromForm() {
  const data = new FormData(form);
  return {
    order_value_ratio: Number(data.get('order_value_ratio')),
    account_age_days: Number(data.get('account_age_days')),
    txns_last_hour: Number(data.get('txns_last_hour')),
    billing_shipping_mismatch: data.has('billing_shipping_mismatch') ? 1 : 0,
    vpn_detected: data.has('vpn_detected') ? 1 : 0,
    new_device: data.has('new_device') ? 1 : 0,
    hour_of_day: Number(data.get('hour_of_day')),
  };
}

function renderScore(data) {
  result.className = `panel result-panel ${data.decision}`;
  result.innerHTML = `<div class="result-content">
    <p class="decision-label">Model assessment</p>
    <div class="risk-number">${data.risk_score}<small>%</small></div>
    <span class="decision">${data.decision}</span>
    <div class="reasons"><h3>Signals behind this decision</h3><ul>${data.reasons.map((reason) => `<li>${reason}</li>`).join('')}</ul></div>
  </div>`;
  document.querySelector('#model-version').textContent = data.model_version;
}

form.addEventListener('submit', async (event) => {
  event.preventDefault(); error.textContent = '';
  const button = form.querySelector('.score-button'); button.disabled = true;
  try {
    const response = await fetch('/score', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(valuesFromForm()) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'The transaction could not be scored.');
    renderScore(data);
  } catch (requestError) { error.textContent = requestError.message; } finally { button.disabled = false; }
});

document.querySelector('#load-example').addEventListener('click', () => {
  const example = { order_value_ratio: 3.2, account_age_days: 4, txns_last_hour: 5, hour_of_day: 3 };
  Object.entries(example).forEach(([name, value]) => { form.elements[name].value = value; });
  ['billing_shipping_mismatch', 'vpn_detected', 'new_device'].forEach((name) => { form.elements[name].checked = true; });
});

fetch('/health').then((response) => response.json()).then((data) => {
  health.classList.toggle('ready', data.status === 'ok' && data.model_loaded);
  health.innerHTML = `<span class="health-dot"></span> ${data.model_loaded ? 'Model ready' : 'Model unavailable'}`;
}).catch(() => { health.innerHTML = '<span class="health-dot"></span> API unavailable'; });
