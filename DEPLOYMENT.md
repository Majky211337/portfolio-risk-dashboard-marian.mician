# Deployment guide — krok za krokom

Tento návod ťa prevedie celým procesom od **lokálneho projektu** až po **live web stránku** na adrese:

```
https://portfolio-risk-dashboard-marian-mician.onrender.com
```

Celý proces trvá **15–20 minút** a nestojí ani cent.

---

## 0. Predpoklady — čo potrebuješ pred začiatkom

| Vec | Ako overiť | Inštalácia |
|---|---|---|
| **Git** | `git --version` → vidíš číslo verzie | [git-scm.com](https://git-scm.com/downloads) |
| **GitHub účet** | Otvoríš [github.com](https://github.com), si prihlásený | Zaregistruj sa zadarmo |
| **Render účet** | Otvoríš [render.com](https://render.com), si prihlásený | Zaregistruj sa cez GitHub (jeden klik) |

> **Tip:** Pri registrácii na Render použi tlačidlo "Sign up with GitHub" — neskôr to ušetrí jeden krok pri spájaní repa.

---

## 1. Príprava projektu (~2 min)

Otvor terminál (PowerShell alebo Git Bash) a prejdi do priečinka projektu:

```bash
cd "C:\Users\micia\Desktop\daily dataset"
```

### 1.1 Nepovinné — vyčisti staré súbory

Tieto súbory sa už nepoužívajú a možno ich pokojne zmazať pred commitom:

```bash
# Stará verzia appky a CSS (boli nahradené _fixed verziami)
rm app.py
rm assets/styles.css
rm assets/styles.txt
```

> Tieto verzie sú aj tak v `.gitignore` mimo dosahu — ale je čistejšie ich nemať v repe.

### 1.2 Skontroluj že kľúčové súbory existujú

```bash
ls render.yaml Procfile runtime.txt requirements.txt README.md
ls .github/workflows/refresh-data.yml
ls docs/screenshots/01-hero-light.png
```

Všetkých 7 musí existovať. Ak nie — napíš mi.

---

## 2. Inicializácia git repa (~3 min)

### 2.1 Vytvor lokálny repo

```bash
git init
git branch -M main
```

### 2.2 Skontroluj čo všetko sa pôjde commitovať

```bash
git status
```

Mal by si vidieť všetky `.py`, `.md`, `.yml`, `.css`, `data/*.csv` súbory **green/Untracked**. **NEMAL** by si vidieť `__pycache__/`, `yf_cache/`, `outputs/*.csv` (sú v `.gitignore`).

### 2.3 Pridaj všetko a urob prvý commit

```bash
git add .
git commit -m "Initial release: Portfolio Risk Dashboard"
```

> Ak ti git povie "Please tell me who you are" — najprv nastav meno + email:
> ```bash
> git config --global user.name "Marián Mičian"
> git config --global user.email "tvoj@email.sk"
> ```
> a `git commit` znova.

---

## 3. Vytvorenie GitHub repa (~3 min)

### 3.1 Vytvor prázdny repo na github.com

1. Otvor <https://github.com/new>
2. **Repository name:** `portfolio-risk-dashboard`
3. **Description:** `Interactive portfolio risk dashboard — Dash + Plotly`
4. **Public** ✓ (potrebné aby ho Render zdarma videl, a aj pre portfólio efekt)
5. **DON'T initialize** s README/license/.gitignore (už ich máš)
6. Klikni **Create repository**

### 3.2 Pripoj lokálny repo a pushni

GitHub ti ukáže príkazy. Skopíruj sekciu **"…or push an existing repository from the command line"** a spusti v termináli. Pre tvoj prípad to bude (nahraď `<your-user>` tvojím GitHub usernamom):

```bash
git remote add origin https://github.com/<your-user>/portfolio-risk-dashboard.git
git push -u origin main
```

> **Authentication:** GitHub od 2021 nepodporuje heslá v termináli. Pri prvom push-i:
> - **Windows:** Otvorí sa Git Credential Manager popup → prihlás sa cez prehliadač
> - Alternatíva: vytvor [Personal Access Token](https://github.com/settings/tokens) (scope `repo`) a použi ho namiesto hesla

### 3.3 Overenie

Otvor `https://github.com/<your-user>/portfolio-risk-dashboard` — mali by si vidieť:
- README s tvojím menom hore
- 4 screenshoty v `docs/screenshots/`
- Všetky Python súbory
- `render.yaml`

---

## 4. Deploy na Render.com (~5 min)

### 4.1 Pripoj GitHub k Renderu (ak si sa registroval iným emailom)

V Render dashboard: **Account → GitHub** → klikni **Connect** → autorizuj.

### 4.2 Vytvor Blueprint Instance

1. Otvor <https://dashboard.render.com/blueprints>
2. Klikni **New Blueprint Instance**
3. **Connect a repository** → vyber `portfolio-risk-dashboard`
4. Render automaticky **prečíta `render.yaml`** a ukáže ti čo ide vytvoriť:
   - Service name: `portfolio-risk-dashboard-marian-mician`
   - Plan: **Free**
   - Region: **Frankfurt**
   - Branch: `main`
5. Klikni **Apply**

### 4.3 Sleduj prvý build

- Render začne build (~2–3 minúty)
- V live logu uvidíš: `pip install -r requirements.txt` → potom `gunicorn app_fixed:server ...`
- Status sa zmení z **Building** → **Deploying** → **Live**

### 4.4 Otvor live aplikáciu

Klikni na URL hore vpravo. Mal by si vidieť:

```
https://portfolio-risk-dashboard-marian-mician.onrender.com
```

> **Prvý load po spaní:** Free Render dyno zaspí po 15 minútach nečinnosti. Prvý hit potom trvá ~30s (cold start), ďalšie sú instantné. Pred ukážkou kolegom/recruiterovi klikni na link 1 min vopred aby bol warm.

---

## 5. Verifikácia že beží správne (~2 min)

Cez prehliadač otvor live URL a otestuj:

- [x] Hero ukazuje **"PORTFOLIO ANALYTICS PROJECT · BY MARIÁN MIČIAN"**
- [x] **12 metric kariet** so sparklinmi sa zobrazujú
- [x] Klikneš **Dark mode** → celá appka prepne tému
- [x] **Portfolio Value chart** sa vykresľuje
- [x] **Monte Carlo Forecast** sekcia ukazuje fan chart
- [x] Klikneš **Copy share link** → URL bar má `?spy=60&gld=15&...`
- [x] V tab-e prehliadača vidíš title `Portfolio Risk Dashboard – Marián Mičian`
- [x] Footer dolu hovorí `Built by Marián Mičian · 2026`

Ak niečo nefunguje — skoč na **Troubleshooting** dolu.

---

## 6. Aktivácia automatického weekly data refresh

Workflow `refresh-data.yml` je v repe ale ešte musí dostať povolenie pushovať späť do repa.

1. GitHub repo → **Settings** → **Actions** → **General**
2. Sekcia **Workflow permissions** dole
3. Vyber **Read and write permissions**
4. Klikni **Save**

Po tom workflow každú sobotu 06:00 UTC:
1. Stiahne fresh ETF dáta z Yahoo Finance
2. Commitne nové CSV ak sa zmenili
3. Render auto-deploy spustí nový build s čerstvými dátami

**Manuálny test:** Actions tab → **Refresh daily ETF data** → **Run workflow** → vyber main → **Run**.

---

## 7. Bonus — vylepšenia po deploy

### 7.1 GitHub topics (objaviteľnosť)

Repo → klik na ⚙ pri "About" → pridaj tagy:
```
dash plotly python portfolio backtesting finance dashboard data-visualization
```

### 7.2 Live demo link v "About" boxe

Repo → ⚙ pri "About" → **Website**:
```
https://portfolio-risk-dashboard-marian-mician.onrender.com
```

Bude sa zobrazovať priamo nad zoznamom súborov.

### 7.3 Update URL v README

`README.md` má placeholder `<your-user>` v "Built by [Marián Mičian](https://github.com/<your-user>)". Nahraď ho svojím GitHub usernamom:

```bash
# Otvor README.md a nahraď <your-user> tvojím usernamom
# Potom:
git add README.md
git commit -m "docs: fix GitHub profile link"
git push
```

### 7.4 Custom doména (nepovinné, zadarmo na Renderi)

Ak máš vlastnú doménu (napr. cez Webglobe / GoDaddy):
1. Render service → **Settings** → **Custom Domain** → **Add**
2. Napíš napr. `portfolio.marianmician.sk`
3. Render ti dá CNAME hodnotu (napr. `portfolio-risk-dashboard-marian-mician.onrender.com`)
4. U svojho DNS providera nastavíš CNAME záznam
5. Render za pár minút vystaví SSL certifikát (zadarmo cez Let's Encrypt)

---

## 8. Ako updateovať app v budúcnosti

Render má **auto-deploy** zapnutý, takže každý push do `main` automaticky spustí redeploy.

Workflow:

```bash
# Spravil si zmenu v kóde
git add .
git commit -m "feat: niečo nové"
git push

# Render za 2-3 minúty vidí nový commit a redeployne automaticky.
# V Render dashboarde vidíš live progress.
```

---

## 9. Troubleshooting

### Build failne na "Could not find a version that satisfies the requirement"

**Príčina:** Render používa Python verziu inú než `runtime.txt`.

**Fix:** Otvor `render.yaml` a sprav `PYTHON_VERSION` shodný s `runtime.txt`. Aktuálne 3.12.7.

### Build failne na pip install yfinance / dash

**Príčina:** Niektoré balíky majú compiled extensions (napr. numpy, pandas).

**Fix:** Render free tier má dostatok pamäte. Ak failne, väčšinou stačí v Render service → **Manual Deploy** → **Clear build cache & deploy**.

### Cold start trvá > 30s

**Normálne.** Free tier dyno spí. Riešenia:
- **Free:** [UptimeRobot](https://uptimerobot.com/) ping každých 14 min → dyno nezaspí
- **Paid:** $7/mes Starter plan → zero cold starts

### "yfinance: nepodarilo sa stiahnuť" v GitHub Actions

Yahoo občas blokuje. Riešenie:
- Spusti workflow ručne neskôr (väčšinou prejde)
- Skript má 3 retries po 3s — väčšinou stačí

### URL `portfolio-risk-dashboard-marian-mician.onrender.com` už existuje

Render service názvy sú **globálne unikátne**. Ak ti kolega "ukradol" rovnaký názov:
- Otvor `render.yaml` → zmeň `name:` na napr. `portfolio-risk-dashboard-marian-mician-2026`
- `git commit -am "infra: rename render service"`
- `git push`
- Render Blueprint → **Apply Blueprint** → nová URL

### Footer / dark mode na produkčnej URL nefunguje

Vyčisti browser cache (Ctrl+Shift+R) — Render občas cacheuje CSS.

### "Permission denied (publickey)" pri `git push`

Použivaš SSH URL ale nemáš nastavený SSH key. Prepni na HTTPS:

```bash
git remote set-url origin https://github.com/<your-user>/portfolio-risk-dashboard.git
git push
```

---

## 10. Cheat sheet — najčastejšie príkazy

| Akcia | Príkaz |
|---|---|
| Lokálne spustenie | `python app_fixed.py` |
| Refresh dát manuálne | `python download_daily_data.py` |
| Status repa | `git status` |
| Pushni zmeny | `git add . && git commit -m "..." && git push` |
| Vidieť build log na Renderi | Dashboard → service → **Logs** tab |
| Manuálny redeploy | Render service → **Manual Deploy** → **Deploy latest commit** |
| Trigger data refresh | GitHub Actions → **Refresh daily ETF data** → **Run workflow** |

---

## Hotovo 🚀

Tvoja appka beží na webe, GitHub repo je verejný, dáta sa updatujú týždenne, každý kto klikne na link uvidí tvoje meno hore aj dolu.

Ak narazíš na problém — napíš mi ktorý krok a aký error, vyriešime.
