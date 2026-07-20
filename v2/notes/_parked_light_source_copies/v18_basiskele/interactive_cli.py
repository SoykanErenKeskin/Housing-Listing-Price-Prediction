"""Interactive arrow-key CLI prompts for V16 training.

Zero third-party deps: Windows uses msvcrt, Unix uses termios/tty.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Any


@dataclass
class ChoiceOption:
    value: Any
    label: str
    help: str


def _read_key() -> str:
    """Return 'up', 'down', 'enter', 'esc', or a character."""
    if sys.platform == "win32":
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x1b":
            return "esc"
        if ch in ("\x00", "\xe0"):
            ch2 = msvcrt.getwch()
            if ch2 == "H":
                return "up"
            if ch2 == "P":
                return "down"
            return "other"
        return ch

    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x1b":
            seq = sys.stdin.read(2)
            if seq == "[A":
                return "up"
            if seq == "[B":
                return "down"
            return "esc"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _clear_lines(n: int) -> None:
    for _ in range(n):
        sys.stdout.write("\033[1A\033[2K")
    sys.stdout.flush()


def ask_choice(
    title: str,
    description: str,
    options: list[ChoiceOption],
    default_index: int = 0,
) -> Any:
    """Arrow-key multiple choice. Returns selected option.value."""
    if not options:
        raise ValueError("options required")
    idx = max(0, min(default_index, len(options) - 1))
    rendered = 0
    label_width = max(len(opt.label) for opt in options)

    def draw(first: bool = False) -> None:
        nonlocal rendered
        if not first and rendered:
            _clear_lines(rendered)
        lines = [
            "",
            f"▶ {title}",
            f"  {description}",
            "  (↑/↓ seç, Enter onayla)",
            "",
        ]
        for i, opt in enumerate(options):
            mark = "❯" if i == idx else " "
            if i == idx and opt.help:
                lines.append(f"  {mark} {opt.label:<{label_width}}  → {opt.help}")
            else:
                lines.append(f"  {mark} {opt.label}")
        text = "\n".join(lines) + "\n"
        sys.stdout.write(text)
        sys.stdout.flush()
        rendered = text.count("\n")

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        # Non-interactive fallback: print options and read number
        print(f"\n{title}\n{description}")
        for i, opt in enumerate(options):
            print(f"  [{i + 1}] {opt.label} — {opt.help}")
        raw = input(f"Seçim [1-{len(options)}] (default {idx + 1}): ").strip()
        if not raw:
            return options[idx].value
        try:
            pick = int(raw) - 1
            return options[max(0, min(pick, len(options) - 1))].value
        except ValueError:
            return options[idx].value

    draw(first=True)
    while True:
        key = _read_key()
        if key == "up":
            idx = (idx - 1) % len(options)
            draw()
        elif key == "down":
            idx = (idx + 1) % len(options)
            draw()
        elif key == "enter":
            sys.stdout.write(f"  ✓ Seçildi: {options[idx].label}\n")
            sys.stdout.flush()
            return options[idx].value
        elif key == "esc":
            sys.stdout.write(f"  ✓ Varsayılan: {options[idx].label}\n")
            sys.stdout.flush()
            return options[idx].value


def ask_text(title: str, description: str, default: str) -> str:
    print(f"\n▶ {title}")
    print(f"  {description}")
    raw = input(f"  [{default}]: ").strip()
    return raw or default


def cli_flag_provided(argv: list[str], *flags: str) -> bool:
    """True if any of the flags appear on the command line."""
    for a in argv:
        for f in flags:
            if a == f or a.startswith(f + "="):
                return True
    return False


def apply_interactive_prompts(args: Any, argv: list[str] | None = None) -> Any:
    """Fill missing interactive settings on argparse Namespace."""
    argv = list(argv if argv is not None else sys.argv[1:])

    # Skip entirely if --no-interactive
    if cli_flag_provided(argv, "--no-interactive"):
        return args

    needed = []
    if not cli_flag_provided(argv, "--out"):
        needed.append("out")
    if not cli_flag_provided(argv, "--demographics-mode"):
        needed.append("demographics-mode")
    if not cli_flag_provided(argv, "--attribute-mode"):
        needed.append("attribute-mode")
    if not cli_flag_provided(argv, "--detail-effect-mode"):
        needed.append("detail-effect-mode")
    if not cli_flag_provided(argv, "--run-attribute-ablation", "--no-run-attribute-ablation"):
        needed.append("run-attribute-ablation")
    if not cli_flag_provided(argv, "--run-demographics-ablation", "--no-run-demographics-ablation"):
        needed.append("run-demographics-ablation")
    if not cli_flag_provided(argv, "--run-detail-effect-ablation", "--no-run-detail-effect-ablation"):
        needed.append("run-detail-effect-ablation")
    if not cli_flag_provided(argv, "--basiskele-specialist-mode"):
        needed.append("basiskele-specialist-mode")
    if not cli_flag_provided(argv, "--basiskele-variance-lift"):
        needed.append("basiskele-variance-lift")
    if not cli_flag_provided(argv, "--basiskele-large-home-regime"):
        needed.append("basiskele-large-home-regime")
    if not cli_flag_provided(argv, "--basiskele-spread-layer"):
        needed.append("basiskele-spread-layer")
    if not cli_flag_provided(argv, "--karamursel-baseline-mode"):
        needed.append("karamursel-baseline-mode")
    if not cli_flag_provided(argv, "--location-feature-mode"):
        needed.append("location-feature-mode")
    if not cli_flag_provided(argv, "--location-scope"):
        needed.append("location-scope")
    if not cli_flag_provided(argv, "--geo-context-mode"):
        needed.append("geo-context-mode")
    if not cli_flag_provided(argv, "--run-location-ablation", "--no-run-location-ablation"):
        needed.append("run-location-ablation")
    if not cli_flag_provided(argv, "--run-basiskele-specialist-ablation", "--no-run-basiskele-specialist-ablation"):
        needed.append("run-basiskele-specialist-ablation")
    if not cli_flag_provided(argv, "--run-v16-regime-ablation", "--no-run-v16-regime-ablation"):
        needed.append("run-v16-regime-ablation")
    if not cli_flag_provided(argv, "--county-expert-min-rows-overrides"):
        needed.append("county-expert-min-rows-overrides")
    if not cli_flag_provided(argv, "--fast"):
        needed.append("fast")
    if not cli_flag_provided(argv, "--target-mode"):
        needed.append("target-mode")
    if not cli_flag_provided(argv, "--county-expert-min-rows"):
        needed.append("county-expert-min-rows")
    if not cli_flag_provided(argv, "--use-trend", "--no-use-trend"):
        needed.append("use-trend")
    if not cli_flag_provided(argv, "--limit-sale") and not cli_flag_provided(argv, "--limit-rental"):
        needed.append("limits")

    if not needed:
        return args

    print("\n════════════════════════════════════════")
    print(" V18 etkileşimli ayar sihirbazı")
    print(" Komut satırında verdiğin -- bayrakları atlanır.")
    print(f" Sorulacak: {', '.join(needed)}")
    print("════════════════════════════════════════")

    # --- out (text only) ---
    if not cli_flag_provided(argv, "--out"):
        args.out = ask_text(
            "Çıktı klasörü (--out)",
            "Raporlar, model bundle ve CSV'ler buraya yazılır.",
            str(getattr(args, "out", "outputs/v16_full")),
        )

    # --- demographics-mode ---
    if not cli_flag_provided(argv, "--demographics-mode"):
        cur = str(getattr(args, "demographics_mode", "safe"))
        opts = [
            ChoiceOption("none", "none", "Demografi feature yok (baseline karşılaştırma)."),
            ChoiceOption("safe", "safe", "Güvenli demografi seti (önerilen final)."),
            ChoiceOption("full", "full", "Safe + ek demografi kolonları (ablation/challenger)."),
        ]
        default_i = next((i for i, o in enumerate(opts) if o.value == cur), 1)
        args.demographics_mode = ask_choice(
            "Demografi modu (--demographics-mode)",
            "Mahalle/ilçe demografi feature'larının ne kadarının modele gireceği.",
            opts,
            default_i,
        )

    # --- attribute-mode ---
    if not cli_flag_provided(argv, "--attribute-mode"):
        cur = str(getattr(args, "attribute_mode", "full"))
        opts = [
            ChoiceOption("none", "none", "Yeni attr_* yok — V12 benzeri özellik seti."),
            ChoiceOption("basic", "basic", "Yaş/kat/site/ısıtma vb. deterministic kalite skorları."),
            ChoiceOption("full", "full", "Basic + fold-safe attr_effect_* premium feature'ları (önerilen)."),
        ]
        default_i = next((i for i, o in enumerate(opts) if o.value == cur), 2)
        args.attribute_mode = ask_choice(
            "Attribute modu (--attribute-mode)",
            "İlan kalite duyarlılığı feature grubu.",
            opts,
            default_i,
        )

    # --- detail-effect-mode ---
    if not cli_flag_provided(argv, "--detail-effect-mode"):
        cur = str(getattr(args, "detail_effect_mode", "group"))
        opts = [
            ChoiceOption("none", "none", "Yeni detail_effect_* yok — V13 benzeri."),
            ChoiceOption("group", "group", "Grup bazlı local detail premium (önerilen default)."),
            ChoiceOption("full", "full", "Grup + tekil detail binary premium (challenger; overfit riski)."),
        ]
        default_i = next((i for i, o in enumerate(opts) if o.value == cur), 1)
        args.detail_effect_mode = ask_choice(
            "Detail effect modu (--detail-effect-mode)",
            "Lokasyon bağlamında front/view/near/out/in premium etkisi.",
            opts,
            default_i,
        )

    # --- run-attribute-ablation ---
    if not cli_flag_provided(argv, "--run-attribute-ablation", "--no-run-attribute-ablation"):
        cur = bool(getattr(args, "run_attribute_ablation", False))
        opts = [
            ChoiceOption(True, "Evet (aç)", "none/basic/full attribute ablation çalıştır (seçili demografi sabit)."),
            ChoiceOption(False, "Hayır (kapalı)", "Sadece seçili attribute-mode ile tek run."),
        ]
        default_i = 0 if cur else 1
        args.run_attribute_ablation = ask_choice(
            "Attribute ablation (--run-attribute-ablation)",
            "Attribute feature'ların gerçekten işe yarayıp yaramadığını kıyaslar.",
            opts,
            default_i,
        )

    # --- run-demographics-ablation ---
    if not cli_flag_provided(argv, "--run-demographics-ablation", "--no-run-demographics-ablation"):
        cur = bool(getattr(args, "run_demographics_ablation", False))
        opts = [
            ChoiceOption(True, "Evet (aç)", "none/safe/full demografi ablation (seçili attribute sabit)."),
            ChoiceOption(False, "Hayır (kapalı)", "Sadece seçili demographics-mode."),
        ]
        default_i = 0 if cur else 1
        args.run_demographics_ablation = ask_choice(
            "Demografi ablation (--run-demographics-ablation)",
            "Demografi katmanının katkısını ayrı ölçer. Full train'de genelde kapalı tutulabilir.",
            opts,
            default_i,
        )

    # --- run-detail-effect-ablation ---
    if not cli_flag_provided(argv, "--run-detail-effect-ablation", "--no-run-detail-effect-ablation"):
        cur = bool(getattr(args, "run_detail_effect_ablation", False))
        opts = [
            ChoiceOption(True, "Evet (aç)", "none/group/full detail-effect ablation (seçili demo+attr sabit)."),
            ChoiceOption(False, "Hayır (kapalı)", "Sadece seçili detail-effect-mode."),
        ]
        default_i = 0 if cur else 1
        args.run_detail_effect_ablation = ask_choice(
            "Detail-effect ablation (--run-detail-effect-ablation)",
            "Detail premium katmanının katkısını kıyaslar. Default full train'de kapalı.",
            opts,
            default_i,
        )

    # --- basiskele-specialist-mode ---
    if not cli_flag_provided(argv, "--basiskele-specialist-mode"):
        cur = str(getattr(args, "basiskele_specialist_mode", "premium_target_stats"))
        opts = [
            ChoiceOption("none", "none", "Başiskele specialist feature yok."),
            ChoiceOption("premium", "premium", "Deterministic premium skor + flag + etkileşim."),
            ChoiceOption("premium_target_stats", "premium_target_stats", "Premium + fold-safe residual target stats (önerilen)."),
        ]
        default_i = next((i for i, o in enumerate(opts) if o.value == cur), 2)
        args.basiskele_specialist_mode = ask_choice(
            "Başiskele specialist (--basiskele-specialist-mode)",
            "County-level premium lift feature seti.",
            opts,
            default_i,
        )

    # --- basiskele-variance-lift ---
    if not cli_flag_provided(argv, "--basiskele-variance-lift"):
        cur = str(getattr(args, "basiskele_variance_lift", "none"))
        opts = [
            ChoiceOption("none", "none", "Variance-lift kapalı (V16 önerilen)."),
            ChoiceOption("conservative", "conservative", "Küçük lambda; R2 artmazsa kapanır (legacy)."),
            ChoiceOption("full", "full", "Geniş lambda aralığı — sadece deney."),
        ]
        default_i = next((i for i, o in enumerate(opts) if o.value == cur), 0)
        args.basiskele_variance_lift = ask_choice(
            "Başiskele variance lift (--basiskele-variance-lift)",
            "V15 legacy; V16'da spread residual tercih edilir.",
            opts,
            default_i,
        )

    # --- basiskele-large-home-regime ---
    if not cli_flag_provided(argv, "--basiskele-large-home-regime"):
        cur = str(getattr(args, "basiskele_large_home_regime", "simple"))
        opts = [
            ChoiceOption("none", "none", "Large_home regime feature yok."),
            ChoiceOption("simple", "simple", "Deterministic large_home features (önerilen)."),
            ChoiceOption("residual", "residual", "Simple + OOF residual delta blend."),
        ]
        default_i = next((i for i, o in enumerate(opts) if o.value == cur), 1)
        args.basiskele_large_home_regime = ask_choice(
            "Başiskele large_home regime (--basiskele-large-home-regime)",
            "200+ m² / 4+1 rejimi için kontrollü feature veya residual.",
            opts,
            default_i,
        )

    # --- basiskele-spread-layer ---
    if not cli_flag_provided(argv, "--basiskele-spread-layer"):
        cur = str(getattr(args, "basiskele_spread_layer", "conservative"))
        opts = [
            ChoiceOption("none", "none", "Spread residual kapalı."),
            ChoiceOption("conservative", "conservative", "Ridge-only OOF spread (önerilen)."),
            ChoiceOption("full", "full", "Ridge + shallow tree spread."),
        ]
        default_i = next((i for i, o in enumerate(opts) if o.value == cur), 1)
        args.basiskele_spread_layer = ask_choice(
            "Başiskele spread layer (--basiskele-spread-layer)",
            "Ucuz/pahalı uç mean-pulling için OOF residual.",
            opts,
            default_i,
        )

    # --- karamursel-baseline-mode ---
    if not cli_flag_provided(argv, "--karamursel-baseline-mode"):
        cur = str(getattr(args, "karamursel_baseline_mode", "none"))
        opts = [
            ChoiceOption("none", "none", "Kapalı (V18 default — V16 ablation'da finali geçmedi)."),
            ChoiceOption("location_age", "location_age", "Fold-safe district×age residual medians."),
        ]
        default_i = next((i for i, o in enumerate(opts) if o.value == cur), 0)
        args.karamursel_baseline_mode = ask_choice(
            "Karamürsel baseline (--karamursel-baseline-mode)",
            "V18'de varsayılan kapalı.",
            opts,
            default_i,
        )

    # --- location-feature-mode ---
    if not cli_flag_provided(argv, "--location-feature-mode"):
        cur = str(getattr(args, "location_feature_mode", "full"))
        opts = [
            ChoiceOption("none", "none", "Location feature yok (V16-like control)."),
            ChoiceOption("basic", "basic", "Ham lat/lon + precision flags."),
            ChoiceOption("geo", "geo", "Basic + centroid/cluster/anchor + geo-context POI/coast/road."),
            ChoiceOption("comparable", "comparable", "Basic + fold-safe emsal stats."),
            ChoiceOption("full", "full", "Geo-context + comparable."),
        ]
        default_i = next((i for i, o in enumerate(opts) if o.value == cur), 4)
        args.location_feature_mode = ask_choice(
            "Location feature modu (--location-feature-mode)",
            "Koordinatın gayrimenkul değerindeki anlamı (deniz/yol/POI/emsal).",
            opts,
            default_i,
        )

    # --- location-scope ---
    if not cli_flag_provided(argv, "--location-scope"):
        cur = str(getattr(args, "location_scope", "basiskele_only"))
        opts = [
            ChoiceOption("basiskele_only", "basiskele_only", "Sadece Başiskele’de location aktif (önerilen)."),
            ChoiceOption("global", "global", "Coverage >= 0.40 olan tüm ilçelerde aktif."),
        ]
        default_i = next((i for i, o in enumerate(opts) if o.value == cur), 0)
        args.location_scope = ask_choice(
            "Location scope (--location-scope)",
            "Location coverage eksik ilçeleri bozmamak için izolasyon.",
            opts,
            default_i,
        )

    # --- geo-context-mode ---
    if not cli_flag_provided(argv, "--geo-context-mode"):
        cur = str(getattr(args, "geo_context_mode", "full"))
        opts = [
            ChoiceOption("none", "none", "POI/coast/road context yok."),
            ChoiceOption("geo_no_poi", "geo_no_poi", "Sadece missing/exact flags."),
            ChoiceOption("geo_with_coast", "geo_with_coast", "Sahil mesafesi + coastal."),
            ChoiceOption("geo_with_poi", "geo_with_poi", "Coast + POI/road distances."),
            ChoiceOption("full", "full", "Tam geo-context."),
        ]
        default_i = next((i for i, o in enumerate(opts) if o.value == cur), 4)
        args.geo_context_mode = ask_choice(
            "Geo-context derinliği (--geo-context-mode)",
            "Offline OSM cache'den coast/POI/road feature'ları.",
            opts,
            default_i,
        )

    # --- run-location-ablation ---
    if not cli_flag_provided(argv, "--run-location-ablation", "--no-run-location-ablation"):
        cur = bool(getattr(args, "run_location_ablation", False))
        opts = [
            ChoiceOption(False, "Hayır (kapalı)", "Tek location mode ile devam."),
            ChoiceOption(True, "Evet (aç)", "basic/coast/poi/full/comparable ablation."),
        ]
        default_i = 1 if cur else 0
        args.run_location_ablation = ask_choice(
            "Location ablation (--run-location-ablation)",
            "Hangi geo-context katmanının lift verdiğini ölç.",
            opts,
            default_i,
        )

    # --- run-basiskele-specialist-ablation ---
    if not cli_flag_provided(argv, "--run-basiskele-specialist-ablation", "--no-run-basiskele-specialist-ablation"):
        cur = bool(getattr(args, "run_basiskele_specialist_ablation", False))
        opts = [
            ChoiceOption(True, "Evet (aç)", "none/premium/premium_target_stats(/variance_lift) ablation."),
            ChoiceOption(False, "Hayır (kapalı)", "Sadece seçili specialist mode."),
        ]
        default_i = 0 if cur else 1
        args.run_basiskele_specialist_ablation = ask_choice(
            "Başiskele specialist ablation (--run-basiskele-specialist-ablation)",
            "Specialist katmanının county lift katkısını kıyaslar.",
            opts,
            default_i,
        )

    # --- run-v16-regime-ablation ---
    if not cli_flag_provided(argv, "--run-v16-regime-ablation", "--no-run-v16-regime-ablation"):
        cur = bool(getattr(args, "run_v16_regime_ablation", False))
        opts = [
            ChoiceOption(True, "Evet (aç)", "control / large_home / spread / karamursel / combined."),
            ChoiceOption(False, "Hayır (kapalı)", "Sadece seçili regime ayarları."),
        ]
        default_i = 0 if cur else 1
        args.run_v16_regime_ablation = ask_choice(
            "V16 regime ablation (--run-v16-regime-ablation)",
            "Rejim bazlı residual/baseline deneyleri.",
            opts,
            default_i,
        )

    # --- county-expert-min-rows-overrides ---
    if not cli_flag_provided(argv, "--county-expert-min-rows-overrides"):
        cur = str(getattr(args, "county_expert_min_rows_overrides", "Karamürsel:180"))
        args.county_expert_min_rows_overrides = ask_text(
            "County expert min_rows override (--county-expert-min-rows-overrides)",
            "Örn. Karamürsel:180 — sadece belirtilen ilçeler için min_rows düşürür.",
            cur,
        )

    # --- fast ---
    if not cli_flag_provided(argv, "--fast"):
        cur = bool(getattr(args, "fast", False))
        opts = [
            ChoiceOption(True, "Evet — hızlı/hafif modeller", "Daha az estimator; smoke/test için."),
            ChoiceOption(False, "Hayır — full model ayarları", "Gerçek final train için önerilen."),
        ]
        default_i = 0 if cur else 1
        args.fast = ask_choice(
            "Fast mode (--fast)",
            "Model karmaşıklığını düşürür; süre kısalır, skor biraz zayıflayabilir.",
            opts,
            default_i,
        )

    # --- target-mode ---
    if not cli_flag_provided(argv, "--target-mode"):
        cur = str(getattr(args, "target_mode", "residual"))
        opts = [
            ChoiceOption("residual", "residual", "Lokasyon baseline üstüne residual öğren (önerilen)."),
            ChoiceOption("log", "log", "log(fiyat) hedefi."),
            ChoiceOption("raw", "raw", "Ham TL/m² hedefi."),
        ]
        default_i = next((i for i, o in enumerate(opts) if o.value == cur), 0)
        args.target_mode = ask_choice(
            "Hedef modu (--target-mode)",
            "Modelin neyi tahmin etmeye çalıştığı.",
            opts,
            default_i,
        )

    # --- county-expert-min-rows ---
    if not cli_flag_provided(argv, "--county-expert-min-rows"):
        cur = int(getattr(args, "county_expert_min_rows", 180))
        opts = [
            ChoiceOption(180, "180 (V16 default)", "Global min 180; Karamürsel override ile uyumlu."),
            ChoiceOption(250, "250 (legacy)", "V15 tarzı daha sıkı expert eşiği."),
        ]
        default_i = 0 if cur <= 180 else 1
        args.county_expert_min_rows = ask_choice(
            "County expert min satır (--county-expert-min-rows)",
            "İlçe uzman modeli için minimum örnek sayısı.",
            opts,
            default_i,
        )

    # --- use-trend ---
    if not cli_flag_provided(argv, "--use-trend", "--no-use-trend"):
        cur = bool(getattr(args, "use_trend", True))
        opts = [
            ChoiceOption(True, "Evet", "Trend sale m² feature'larını kullan."),
            ChoiceOption(False, "Hayır", "Trend feature'larını kapat."),
        ]
        default_i = 0 if cur else 1
        args.use_trend = ask_choice(
            "Trend feature (--use-trend)",
            "Piyasa trend sinyali (app-safe aggregate).",
            opts,
            default_i,
        )

    # --- data size / limits ---
    if not cli_flag_provided(argv, "--limit-sale") and not cli_flag_provided(argv, "--limit-rental"):
        opts = [
            ChoiceOption("full", "Tam veri", "Limit yok — full train."),
            ChoiceOption("smoke", "Hızlı test (800/800)", "sale+rental 800 satır limit — smoke."),
        ]
        pick = ask_choice(
            "Veri boyutu",
            "Eğitimde kullanılacak satır limiti.",
            opts,
            0,
        )
        if pick == "smoke":
            args.limit_sale = 800
            args.limit_rental = 800
        else:
            args.limit_sale = None
            args.limit_rental = None

    print("\n── Özet ──")
    print(f"  out={args.out}")
    print(f"  demographics-mode={args.demographics_mode}")
    print(f"  attribute-mode={args.attribute_mode}")
    print(f"  detail-effect-mode={getattr(args, 'detail_effect_mode', 'group')}")
    print(f"  basiskele-specialist-mode={getattr(args, 'basiskele_specialist_mode', 'premium_target_stats')}")
    print(f"  basiskele-variance-lift={getattr(args, 'basiskele_variance_lift', 'none')}")
    print(f"  location-feature-mode={getattr(args, 'location_feature_mode', 'full')}")
    print(f"  location-scope={getattr(args, 'location_scope', 'basiskele_only')}")
    print(f"  geo-context-mode={getattr(args, 'geo_context_mode', 'full')}")
    print(f"  basiskele-large-home-regime={getattr(args, 'basiskele_large_home_regime', 'none')}")
    print(f"  basiskele-spread-layer={getattr(args, 'basiskele_spread_layer', 'none')}")
    print(f"  karamursel-baseline-mode={getattr(args, 'karamursel_baseline_mode', 'none')}")
    print(f"  county-expert-min-rows-overrides={getattr(args, 'county_expert_min_rows_overrides', 'Karamürsel:180')}")
    print(f"  run-attribute-ablation={args.run_attribute_ablation}")
    print(f"  run-demographics-ablation={args.run_demographics_ablation}")
    print(f"  run-detail-effect-ablation={getattr(args, 'run_detail_effect_ablation', False)}")
    print(f"  run-location-ablation={getattr(args, 'run_location_ablation', False)}")
    print(f"  run-basiskele-specialist-ablation={getattr(args, 'run_basiskele_specialist_ablation', False)}")
    print(f"  run-v16-regime-ablation={getattr(args, 'run_v16_regime_ablation', False)}")
    print(f"  fast={args.fast}")
    print(f"  target-mode={args.target_mode}")
    print(f"  county-expert-min-rows={args.county_expert_min_rows}")
    print(f"  use-trend={args.use_trend}")
    print(f"  limit-sale={args.limit_sale}  limit-rental={args.limit_rental}")
    print("──────────\n")
    return args


def build_early_arg_parser() -> Any:
    """Argparse without importing the heavy training module."""
    import argparse

    ap = argparse.ArgumentParser(
        description="V16 attribute-sensitivity training pipeline. "
        "Eksik ana ayarlar terminalde ok tuşlarıyla sorulur; --no-interactive ile kapat."
    )
    ap.add_argument("--db-url", default=None)
    ap.add_argument("--sale-table", default="sale_listings")
    ap.add_argument("--rental-table", default="rental_listings")
    ap.add_argument("--trend-table", default="trend_observed")
    ap.add_argument("--city", default="Kocaeli")
    ap.add_argument("--counties", default="Başiskele")
    ap.add_argument("--out", default="outputs/v17_full")
    ap.add_argument("--target-mode", choices=["residual", "log", "raw"], default="residual")
    ap.add_argument("--models", default="ridge,gradient_boosting,extra_trees,random_forest")
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--use-trend", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--trend-max-date", default=None)
    ap.add_argument("--limit-sale", type=int, default=None)
    ap.add_argument("--limit-rental", type=int, default=None)
    ap.add_argument("--sale-json", default=None)
    ap.add_argument("--rental-json", default=None)
    ap.add_argument("--min-sale-unit-price", type=float, default=8_000)
    ap.add_argument("--max-sale-unit-price", type=float, default=200_000)
    ap.add_argument("--min-rent-m2", type=float, default=50)
    ap.add_argument("--max-rent-m2", type=float, default=2_500)
    ap.add_argument("--location-outlier-filter", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--min-location-ratio", type=float, default=0.50)
    ap.add_argument("--max-location-ratio", type=float, default=1.90)
    ap.add_argument("--location-mad-threshold", type=float, default=3.50)
    ap.add_argument("--location-min-group-size", type=int, default=12)
    ap.add_argument("--county-experts", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--county-expert-min-rows", type=int, default=180)
    ap.add_argument("--anomaly-reports", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--demographics-table", default="district_demographics")
    ap.add_argument("--demographics-mode", choices=["none", "safe", "full"], default="safe")
    ap.add_argument("--exclude-anomalies-threshold", type=float, default=25.0)
    ap.add_argument("--attribute-mode", choices=["none", "basic", "full"], default="full")
    ap.add_argument("--detail-effect-mode", choices=["none", "group", "full"], default="group")
    ap.add_argument("--run-attribute-ablation", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--run-demographics-ablation", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--run-detail-effect-ablation", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--county-expert-min-rows-overrides", default="Karamürsel:180")
    ap.add_argument(
        "--basiskele-specialist-mode",
        choices=["none", "premium", "premium_target_stats", "premium_target_stats_variance_lift"],
        default="premium_target_stats",
    )
    ap.add_argument("--basiskele-variance-lift", choices=["none", "conservative", "full"], default="none")
    ap.add_argument("--large-home-specialist-mode", choices=["legacy", "redesigned"], default="redesigned")
    ap.add_argument("--basiskele-large-home-regime", choices=["none", "simple", "residual"], default="none")
    ap.add_argument("--basiskele-spread-layer", choices=["none", "conservative", "full"], default="none")
    ap.add_argument("--karamursel-baseline-mode", choices=["none", "location_age"], default="none")
    ap.add_argument("--location-feature-mode", choices=["none", "basic", "geo"], default="geo")
    ap.add_argument("--location-scope", choices=["global", "basiskele_only"], default="basiskele_only")
    ap.add_argument("--location-coverage-min", type=float, default=0.40)
    ap.add_argument(
        "--geo-context-mode",
        choices=["none", "geo_with_coast", "geo_no_poi", "geo_with_poi", "full"],
        default="geo_with_coast",
    )
    ap.add_argument("--location-min-precision", choices=["exact_map", "approx_map", "any"], default="any")
    ap.add_argument("--enable-coordinate-noise-check", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--model-scope", choices=["basiskele_only"], default="basiskele_only")
    ap.add_argument(
        "--comparable-mode",
        choices=["none", "nearest", "similar", "weighted", "large_home", "full"],
        default="full",
    )
    ap.add_argument("--run-comparable-ablation", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--comparable-k-list", default="5,10,20")
    ap.add_argument("--geo-context-cache-dir", default="data/external/geo_context")
    ap.add_argument("--run-location-ablation", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--run-basiskele-specialist-ablation", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--run-v16-regime-ablation", action=argparse.BooleanOptionalAction, default=False)
    ap.add_argument("--interactive", action=argparse.BooleanOptionalAction, default=True)
    return ap


def parse_cli_early(argv: list[str] | None = None) -> Any:
    """Parse CLI + interactive prompts BEFORE heavy ML imports."""
    argv = list(argv if argv is not None else sys.argv[1:])
    ap = build_early_arg_parser()
    args = ap.parse_args(argv)
    if bool(getattr(args, "interactive", True)):
        args = apply_interactive_prompts(args, argv)
    return args
