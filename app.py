import marimo

__generated_with = "0.23.0"
app = marimo.App(
    width="full",
    app_title="How to Choose the Right Predictive Model",
)


@app.cell
def imports():
    import marimo as mo

    import altair as alt
    import numpy as np
    import pandas as pd

    _ = alt.data_transformers.disable_max_rows()
    return alt, mo, np, pd


@app.cell
def theme():
    train_color = "#2A6F97"
    valid_color = "#C9791B"
    test_color = "#353B45"
    truth_color = "#111111"
    chosen_color = "#3F7D52"
    residual_color = "#C65F3B"
    mse_color = "#4C6F91"
    bias_color = "#B75A3C"
    variance_color = "#5E7D50"
    split_a_color = "#C9791B"
    split_b_color = "#6C8BA4"
    cv_color = chosen_color

    style = """
    <style>
    html, body, div, span, p, li, label, button, input, select, textarea, h1, h2, h3, h4 {
      font-family: "Segoe UI", "Trebuchet MS", "DejaVu Sans", Arial, sans-serif !important;
      text-align: left !important;
    }

    .markdown p,
    .markdown li,
    .markdown h1,
    .markdown h2,
    .markdown h3,
    .markdown h4 {
      text-align: left !important;
    }

    .markdown h1,
    .markdown h2,
    .markdown h3,
    .markdown h4 {
      margin-top: 0.04rem !important;
      margin-bottom: 0.04rem !important;
      line-height: 1.2 !important;
    }

    .markdown h1 {
      font-size: 2.05rem !important;
      font-weight: 650 !important;
    }

    .markdown p,
    .markdown ul,
    .markdown ol {
      margin-top: 0.04rem !important;
      margin-bottom: 0.12rem !important;
      line-height: 1.45 !important;
    }

    .markdown ul,
    .markdown ol {
      padding-left: 1.15rem !important;
    }

    .results-panel {
      display: grid;
      gap: 0.34rem;
      margin-top: 0.04rem;
    }

    .results-card {
      border: 1px solid #D9E1E7;
      border-radius: 10px;
      padding: 0.55rem 0.65rem;
      background: #FAFCFE;
    }

    .results-heading {
      font-size: 0.84rem;
      font-weight: 600;
      color: #5D6B78;
      margin-bottom: 0.35rem;
      letter-spacing: 0.01em;
    }

    .results-row {
      display: flex;
      justify-content: space-between;
      gap: 0.75rem;
      align-items: baseline;
      margin: 0.12rem 0;
    }

    .results-row + .results-row {
      border-top: 1px solid #EDF2F6;
      padding-top: 0.22rem;
    }

    .results-label {
      color: #5D6B78;
      font-size: 0.84rem;
    }

    .results-value {
      color: #1F2A36;
      font-weight: 600;
      font-size: 0.96rem;
      text-align: right;
    }
    </style>
    """
    return (
        bias_color,
        chosen_color,
        cv_color,
        mse_color,
        residual_color,
        split_a_color,
        split_b_color,
        style,
        test_color,
        train_color,
        truth_color,
        valid_color,
        variance_color,
    )


@app.cell
def lecture_config():
    age_domain = (18.0, 75.0)
    y_domain = (4.0, 8.5)
    plot_domain = (3.95, 8.55)
    degrees = tuple(range(0, 11))
    return age_domain, degrees, plot_domain, y_domain


@app.cell
def sampling_helpers(age_domain, np):
    def age_from_unit(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return age_domain[0] + x * (age_domain[1] - age_domain[0])

    def true_function(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return 7.35 - 5.7 * x + 6.8 * x**2

    def generate_points(
        n_samples: int,
        noise_std: float,
        seed: int,
        x_low: float = 0.06,
        x_high: float = 0.94,
    ) -> tuple[np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        x = np.sort(rng.uniform(x_low, x_high, n_samples))
        y = true_function(x) + rng.normal(0.0, noise_std, n_samples)
        return x, y

    return age_from_unit, generate_points, true_function


@app.cell
def polynomial_helpers(np, plot_domain):
    def polynomial_matrix(x: np.ndarray, degree: int) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        return np.column_stack([x**power for power in range(degree + 1)])

    def fit_polynomial(x: np.ndarray, y: np.ndarray, degree: int) -> dict[str, np.ndarray | int]:
        design = polynomial_matrix(x, degree)
        coef, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
        return {"degree": degree, "coef": coef}

    def predict_polynomial(model: dict[str, np.ndarray | int], x: np.ndarray) -> np.ndarray:
        design = polynomial_matrix(np.asarray(x, dtype=float), int(model["degree"]))
        return design @ np.asarray(model["coef"], dtype=float)

    def clipped_prediction(model: dict[str, np.ndarray | int], x: np.ndarray) -> np.ndarray:
        return np.clip(predict_polynomial(model, x), plot_domain[0], plot_domain[1])

    def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)
        return float(np.mean((y_true - y_pred) ** 2))

    return clipped_prediction, fit_polynomial, mse, predict_polynomial


@app.cell
def resampling_helpers(np):
    def split_indices(
        n_samples: int,
        train_frac: float,
        valid_frac: float,
        seed: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_samples)
        n_train = int(np.floor(n_samples * train_frac))
        n_valid = int(np.floor(n_samples * valid_frac))
        train_idx = np.sort(order[:n_train])
        valid_idx = np.sort(order[n_train:n_train + n_valid])
        test_idx = np.sort(order[n_train + n_valid:])
        return train_idx, valid_idx, test_idx

    def kfold_indices(n_samples: int, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(seed)
        order = rng.permutation(n_samples)
        folds = np.array_split(order, n_splits)
        rows: list[tuple[np.ndarray, np.ndarray]] = []
        for fold_index in range(n_splits):
            valid_idx = np.sort(folds[fold_index])
            train_idx = np.sort(np.concatenate([fold for idx, fold in enumerate(folds) if idx != fold_index]))
            rows.append((train_idx, valid_idx))
        return rows

    return kfold_indices, split_indices


@app.cell
def selection_helpers(
    fit_polynomial,
    generate_points,
    kfold_indices,
    mse,
    np,
    pd,
    predict_polynomial,
    split_indices,
):
    def evaluate_degree_curves(
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_new: np.ndarray,
        y_new: np.ndarray,
        degree_grid: tuple[int, ...],
    ) -> pd.DataFrame:
        rows: list[dict[str, float | int | str]] = []
        for degree in degree_grid:
            model = fit_polynomial(x_train, y_train, degree)
            rows.append(
                {
                    "Degree": degree,
                    "Dataset": "Training",
                    "MSE": mse(y_train, predict_polynomial(model, x_train)),
                }
            )
            rows.append(
                {
                    "Degree": degree,
                    "Dataset": "Unseen data",
                    "MSE": mse(y_new, predict_polynomial(model, x_new)),
                }
            )
        return pd.DataFrame(rows)

    def build_selection_data(
        seed: int,
        development_n: int = 60,
        test_n: int = 26,
        noise_std: float = 0.48,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        x_dev, y_dev = generate_points(n_samples=development_n, noise_std=noise_std, seed=seed, x_low=0.03, x_high=0.97)
        x_test, y_test = generate_points(n_samples=test_n, noise_std=noise_std, seed=seed + 41, x_low=0.03, x_high=0.97)
        return x_dev, y_dev, x_test, y_test

    def evaluate_validation_curves(
        x_dev: np.ndarray,
        y_dev: np.ndarray,
        degree_grid: tuple[int, ...],
        split_seed: int,
        train_frac: float = 0.70,
    ) -> tuple[pd.DataFrame, dict[str, float | int]]:
        train_idx, valid_idx, _ = split_indices(len(x_dev), train_frac=train_frac, valid_frac=1.0 - train_frac, seed=split_seed)
        rows: list[dict[str, float | int | str]] = []
        for degree in degree_grid:
            model = fit_polynomial(x_dev[train_idx], y_dev[train_idx], degree)
            rows.append(
                {
                    "Degree": degree,
                    "Dataset": "Training",
                    "MSE": mse(y_dev[train_idx], predict_polynomial(model, x_dev[train_idx])),
                }
            )
            rows.append(
                {
                    "Degree": degree,
                    "Dataset": "Validation",
                    "MSE": mse(y_dev[valid_idx], predict_polynomial(model, x_dev[valid_idx])),
                }
            )
        frame = pd.DataFrame(rows)
        validation_rows = frame[frame["Dataset"] == "Validation"].reset_index(drop=True)
        chosen_degree = int(validation_rows.loc[validation_rows["MSE"].idxmin(), "Degree"])
        summary = {
            "chosen_degree": chosen_degree,
            "chosen_validation_mse": float(
                validation_rows.loc[validation_rows["Degree"] == chosen_degree, "MSE"].iloc[0]
            ),
            "train_count": int(len(train_idx)),
            "validation_count": int(len(valid_idx)),
            "train_frac": float(train_frac),
        }
        return frame, summary

    def evaluate_cv_curves(
        x_dev: np.ndarray,
        y_dev: np.ndarray,
        degree_grid: tuple[int, ...],
        n_splits: int,
        seed: int,
    ) -> pd.DataFrame:
        rows: list[dict[str, float | int]] = []
        for degree in degree_grid:
            fold_mses = []
            for train_idx, valid_idx in kfold_indices(len(x_dev), n_splits=n_splits, seed=seed):
                model = fit_polynomial(x_dev[train_idx], y_dev[train_idx], degree)
                fold_mses.append(mse(y_dev[valid_idx], predict_polynomial(model, x_dev[valid_idx])))
            rows.append({"Degree": degree, "CV MSE": float(np.mean(fold_mses))})
        return pd.DataFrame(rows)

    def cap_mse_frame(
        frame: pd.DataFrame,
        field: str = "MSE",
        display_field: str = "Display MSE",
        quantile: float = 0.90,
        floor: float = 1.0,
    ) -> pd.DataFrame:
        plot_frame = frame.copy()
        cap = max(floor, float(plot_frame[field].quantile(quantile)) + 0.20)
        plot_frame[display_field] = np.minimum(plot_frame[field], cap)
        return plot_frame

    return (
        build_selection_data,
        cap_mse_frame,
        evaluate_cv_curves,
        evaluate_degree_curves,
        evaluate_validation_curves,
    )


@app.cell
def bias_variance_helpers(
    fit_polynomial,
    generate_points,
    mse,
    np,
    pd,
    plot_domain,
    predict_polynomial,
    true_function,
):
    def bias_variance_summary(
        degree: int,
        seed: int,
        n_resamples: int = 7,
        sample_size: int = 11,
        noise_std: float = 0.58,
    ) -> dict[str, np.ndarray | float]:
        x_grid = np.linspace(0.0, 1.0, 240)
        predictions = []
        for offset in range(n_resamples):
            x_sample, y_sample = generate_points(sample_size, noise_std=noise_std, seed=seed + 11 * offset)
            model = fit_polynomial(x_sample, y_sample, degree)
            predictions.append(predict_polynomial(model, x_grid))
        prediction_array = np.asarray(predictions)
        truth = true_function(x_grid)
        average = prediction_array.mean(axis=0)
        return {
            "x_grid": x_grid,
            "truth": truth,
            "predictions": prediction_array,
            "plot_predictions": np.clip(prediction_array, plot_domain[0], plot_domain[1]),
            "average": average,
            "bias_sq": float(np.mean((average - truth) ** 2)),
            "variance": float(np.mean(np.var(prediction_array, axis=0))),
            "expected_mse": float(np.mean((prediction_array - truth[None, :]) ** 2) + noise_std**2),
            "noise_var": float(noise_std**2),
        }

    def evaluate_bias_variance_curves(
        seed: int,
        n_resamples: int,
        degree_grid: tuple[int, ...],
        sample_size: int = 25,
        noise_std: float = 0.58,
        x_grid_points: int = 320,
    ) -> pd.DataFrame:
        rows: list[dict[str, float | int | str]] = []
        x_grid = np.linspace(0.0, 1.0, x_grid_points)
        truth = true_function(x_grid)
        population_x = np.linspace(0.0, 1.0, 800)
        population_truth = true_function(population_x)
        for degree in degree_grid:
            population_model = fit_polynomial(population_x, population_truth, degree)
            population_prediction = predict_polynomial(population_model, x_grid)
            bias_sq = mse(truth, population_prediction)
            predictions = []
            for offset in range(n_resamples):
                x_sample, y_sample = generate_points(
                    sample_size,
                    noise_std=noise_std,
                    seed=seed + degree * 100 + 13 * offset,
                )
                model = fit_polynomial(x_sample, y_sample, degree)
                predictions.append(predict_polynomial(model, x_grid))
            prediction_array = np.asarray(predictions)
            variance = float(np.mean(np.var(prediction_array, axis=0)))
            expected_mse = float(bias_sq + variance + noise_std**2)
            rows.extend(
                [
                    {"Degree": degree, "Metric": "E[MSE]", "Value": expected_mse},
                    {"Degree": degree, "Metric": "Bias²", "Value": bias_sq},
                    {"Degree": degree, "Metric": "Variance", "Value": variance},
                ]
            )
        return pd.DataFrame(rows)

    return bias_variance_summary, evaluate_bias_variance_curves


@app.cell
def ui_helpers(alt, mo):
    def finish_chart(chart: alt.Chart, width: int = 360, height: int = 245) -> alt.Chart:
        return (
            chart.properties(width=width, height=height)
            .configure_view(stroke=None)
            .configure_axis(
                labelColor="#2F3441",
                titleColor="#2F3441",
                gridColor="#D9E1E7",
                domainColor="#A7B3BF",
                tickColor="#A7B3BF",
            )
            .configure_legend(labelColor="#2F3441", titleColor="#2F3441")
        )

    def counter_button(label: str, kind: str):
        return mo.ui.button(
            value=0,
            on_click=lambda value: value + 1,
            label=label,
            kind=kind,
            full_width=True,
        )

    def two_col(main: object, side: object):
        return mo.hstack([main, side], widths=[0.95, 1.8], gap=0.55, align="start", wrap=True)

    def section_md(title: str, kicker: str, body: str = ""):
        blocks = [mo.md(f"## {title}"), mo.md(kicker)]
        if body:
            blocks.append(mo.md(body))
        return mo.vstack(blocks, gap=0.03)

    def metrics_md(metrics: list[tuple[str, str]]):
        if not metrics:
            return mo.md("")
        rows = "".join(
            f"<div class='results-row'><div class='results-label'>{label}</div><div class='results-value'>{value}</div></div>"
            for label, value in metrics
        )
        return mo.Html(f"<div class='results-panel'><div class='results-card'><div class='results-heading'>Results</div>{rows}</div></div>")

    def table_md(headers: list[str], rows: list[list[str]]):
        if not rows:
            return mo.md("")
        head = "| " + " | ".join(headers) + " |"
        rule = "| " + " | ".join(["---"] * len(headers)) + " |"
        body = "\n".join("| " + " | ".join(row) + " |" for row in rows)
        return mo.md(f"**Results**\n\n{head}\n{rule}\n{body}")

    def takeaway_md(text: str):
        return mo.md(f"**Takeaway.** {text}")

    def note_md(text: str):
        return mo.md(f"_{text}_")

    def questions_md(questions: list[str]):
        items = "\n".join(f"- {question}" for question in questions)
        return mo.vstack([mo.md("**Make sure you can answer these questions**"), mo.md(items)], gap=0.02)

    def sidebar(
        widgets: list[object],
        metrics: list[tuple[str, str]],
    ):
        blocks: list[object] = [mo.md("**Controls**"), *widgets]
        if metrics:
            blocks.append(metrics_md(metrics))
        return mo.vstack(blocks, gap=0.18, align="stretch")

    return (
        counter_button,
        finish_chart,
        note_md,
        questions_md,
        section_md,
        sidebar,
        takeaway_md,
        two_col,
    )


@app.cell
def intro(mo, style):
    _page = mo.vstack(
        [
            mo.Html(style),
            mo.md(
                "# How to Choose the Right Predictive Model"
            ),
            mo.md(
                "*Model complexity, bias-variance, and cross-validation*"
            ),
            mo.md(
                "This app builds intuition about **generalization**: why training MSE is not enough, how overfitting appears, and why validation and cross-validation help us choose model complexity."
            ),
            mo.md(
                "How to use it: move the sliders to change the polynomial model, press the `Recreate` buttons to draw fresh random data, and use the reveal switches only after making a prediction yourself. Each section is independent, so you can explore one idea without losing your place in the rest of the app."
            ),
        ],
        gap=0.30,
    )
    _page
    return


@app.cell
def s1_controls(counter_button, mo):
    s1_degree = mo.ui.slider(0, 10, value=3, step=1, label="Polynomial degree")
    s1_recreate = counter_button(label="Recreate data", kind="success")
    return s1_degree, s1_recreate


@app.cell
def s1_section(
    age_domain,
    age_from_unit,
    alt,
    chosen_color,
    clipped_prediction,
    finish_chart,
    fit_polynomial,
    generate_points,
    mo,
    mse,
    note_md,
    np,
    pd,
    predict_polynomial,
    questions_md,
    residual_color,
    s1_degree,
    s1_recreate,
    section_md,
    sidebar,
    takeaway_md,
    train_color,
    two_col,
    y_domain,
):
    _seed = 101 + int(s1_recreate.value or 0)
    _x_train, _y_train = generate_points(n_samples=11, noise_std=0.42, seed=_seed)
    _model = fit_polynomial(_x_train, _y_train, int(s1_degree.value))
    _x_grid = np.linspace(0.0, 1.0, 260)

    _point_frame = pd.DataFrame(
        {
            "Age": age_from_unit(_x_train),
            "Observed": _y_train,
            "Predicted": clipped_prediction(_model, _x_train),
            "Residual": _y_train - predict_polynomial(_model, _x_train),
        }
    )
    _line_frame = pd.DataFrame(
        {
            "Age": age_from_unit(_x_grid),
            "Prediction": clipped_prediction(_model, _x_grid),
        }
    )

    _residuals = alt.Chart(_point_frame).mark_rule(
        color=residual_color,
        opacity=0.55,
        strokeWidth=2.0,
    ).encode(
        x=alt.X("Age:Q", scale=alt.Scale(domain=list(age_domain)), title="Age"),
        y=alt.Y("Observed:Q", scale=alt.Scale(domain=list(y_domain)), title="Life satisfaction"),
        y2="Predicted:Q",
    )
    _points = alt.Chart(_point_frame).mark_circle(
        color=train_color,
        size=78,
        opacity=0.93,
    ).encode(
        x="Age:Q",
        y="Observed:Q",
        tooltip=[
            alt.Tooltip("Age:Q", format=".1f"),
            alt.Tooltip("Observed:Q", format=".2f"),
            alt.Tooltip("Predicted:Q", format=".2f"),
            alt.Tooltip("Residual:Q", format=".2f"),
        ],
    )
    _fit_line = alt.Chart(_line_frame).mark_line(
        color=chosen_color,
        strokeWidth=3.0,
        clip=True,
    ).encode(
        x="Age:Q",
        y=alt.Y("Prediction:Q", scale=alt.Scale(domain=list(y_domain))),
    )
    _chart = finish_chart(_residuals + _points + _fit_line)

    _sidebar = sidebar(
        widgets=[s1_degree, s1_recreate],
        metrics=[("Training MSE", f"{mse(_y_train, predict_polynomial(_model, _x_train)):.2f}")],
    )
    _layout = mo.vstack(
        [
            section_md(
                "1. Measuring Fit with Training MSE",
                "We want to understand the relationship between age and life satisfaction.",
                (
                    "Before we compare models, we need a metric that tells us how well a model fits observed data. In this section that metric is **mean squared error (MSE)**. \n"
                    r"$\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat y_i)^2$"
                    "\n\n"
                    "Use the polynomial-degree slider to change the fitted relationship between age and life satisfaction, and use `Recreate data` to draw a fresh sample from the same population."
                ),
            ),

            note_md(r"The orange segments are residuals $(y_i - \hat y_i)$. MSE squares them before averaging."),
            two_col(_sidebar, _chart),
            note_md(
                "Note that very flexible models can pass close to almost every training point. That makes training MSE look excellent even before we ask whether the model will generalize."
            ),
            takeaway_md("Training MSE only tells you how well the model matches this sample. It does not yet answer the generalization question."),
            questions_md(
                [
                    "Why do large residuals matter more in MSE than small residuals?",
                    "Why could a lower training MSE still be misleading for model choice?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s2_controls(counter_button, mo):
    s2_degree = mo.ui.slider(0, 10, value=8, step=1, label="Polynomial degree")
    s2_recreate = counter_button(label="Recreate data", kind="success")
    s2_reveal_test = mo.ui.switch(value=False, label="Reveal test data")
    s2_show_train = mo.ui.switch(value=True, label="Show training points")
    return s2_degree, s2_recreate, s2_reveal_test, s2_show_train


@app.cell
def s2_section(
    age_domain,
    age_from_unit,
    alt,
    chosen_color,
    clipped_prediction,
    finish_chart,
    fit_polynomial,
    generate_points,
    mo,
    mse,
    note_md,
    np,
    pd,
    predict_polynomial,
    questions_md,
    s2_degree,
    s2_recreate,
    s2_reveal_test,
    s2_show_train,
    section_md,
    sidebar,
    takeaway_md,
    test_color,
    train_color,
    two_col,
    y_domain,
):
    _seed = 150 + int(s2_recreate.value or 0)
    _x_train, _y_train = generate_points(n_samples=11, noise_std=0.48, seed=_seed)
    _model = fit_polynomial(_x_train, _y_train, int(s2_degree.value))
    _x_grid = np.linspace(0.0, 1.0, 260)
    _show_test = bool(s2_reveal_test.value)
    _show_train = bool(s2_show_train.value)

    _layers = []
    _fit_frame = pd.DataFrame(
        {
            "Age": age_from_unit(_x_grid),
            "Prediction": clipped_prediction(_model, _x_grid),
        }
    )
    _layers.append(
        alt.Chart(_fit_frame).mark_line(color=chosen_color, strokeWidth=3.0, clip=True).encode(
            x=alt.X("Age:Q", scale=alt.Scale(domain=list(age_domain)), title="Age"),
            y=alt.Y("Prediction:Q", scale=alt.Scale(domain=list(y_domain)), title="Life satisfaction"),
        )
    )

    if _show_train:
        _train_frame = pd.DataFrame({"Age": age_from_unit(_x_train), "Outcome": _y_train})
        _layers.append(
            alt.Chart(_train_frame).mark_circle(color=train_color, size=72, opacity=0.83).encode(
                x="Age:Q",
                y=alt.Y("Outcome:Q", scale=alt.Scale(domain=list(y_domain))),
                tooltip=[alt.Tooltip("Age:Q", format=".1f"), alt.Tooltip("Outcome:Q", format=".2f")],
            )
        )

    _metrics = [("Training MSE", f"{mse(_y_train, predict_polynomial(_model, _x_train)):.2f}")]
    if _show_test:
        _x_test, _y_test = generate_points(
            n_samples=8,
            noise_std=0.56,
            seed=_seed + 700,
            x_low=0.08,
            x_high=0.92,
        )
        _test_frame = pd.DataFrame({"Age": age_from_unit(_x_test), "Outcome": _y_test})
        _layers.append(
            alt.Chart(_test_frame).mark_point(
                shape="diamond",
                filled=True,
                color=test_color,
                size=160,
                opacity=0.95,
            ).encode(
                x="Age:Q",
                y=alt.Y("Outcome:Q", scale=alt.Scale(domain=list(y_domain))),
                tooltip=[alt.Tooltip("Age:Q", format=".1f"), alt.Tooltip("Outcome:Q", format=".2f")],
            )
        )
        _metrics.append(("Test MSE", f"{mse(_y_test, predict_polynomial(_model, _x_test)):.2f}"))

    _chart = finish_chart(alt.layer(*_layers))
    _sidebar = sidebar(
        widgets=[s2_degree, s2_recreate, s2_reveal_test, s2_show_train],
        metrics=_metrics,
    )
    _layout = mo.vstack(
        [
            section_md(
                "2. From Training Data to Test (unseen) Data",
                "To estimate generalization error, training data is not enough. We need unseen data (test data).",
                (
                    "What we really care about is the **generalization error**: how well the model performs on new, unseen data. We estimate generalization error on new data, not on the same data used to fit the model.\n\n"
                    "In practice, we often split observed data into training data and test data. The line below is fitted using the training data only. Then the test-data switch reveals fresh points from the same population without changing the fitted line."
                ),
            ),
            two_col(_sidebar, _chart),
            note_md("The fitted line does not move when the test points appear. The new points are only for evaluation, not for fitting."),
            takeaway_md("Training MSE answers how well the model fits the data it already saw. To estimate generalization error, we need new data."),
            questions_md(
                [
                    "Before revealing the test data, which degrees do you expect to generalize best?",
                    "After the unseen points appear, what signs suggest overfitting?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s3_controls(counter_button, mo):
    s3_degree = mo.ui.slider(0, 10, value=8, step=1, label="Highlighted degree")
    s3_recreate = counter_button(label="Recreate data", kind="success")
    return s3_degree, s3_recreate


@app.cell
def s3_section(
    alt,
    chosen_color,
    degrees,
    evaluate_degree_curves,
    finish_chart,
    generate_points,
    mo,
    note_md,
    np,
    pd,
    questions_md,
    s3_degree,
    s3_recreate,
    section_md,
    sidebar,
    takeaway_md,
    test_color,
    train_color,
    two_col,
):
    _seed = 210 + int(s3_recreate.value or 0)
    _x_train, _y_train = generate_points(n_samples=11, noise_std=0.56, seed=_seed)
    _x_new, _y_new = generate_points(n_samples=60, noise_std=0.56, seed=_seed + 31, x_low=0.02, x_high=0.98)
    _curve_frame = evaluate_degree_curves(_x_train, _y_train, _x_new, _y_new, degrees)
    _plot_frame = _curve_frame.copy()
    _plot_frame["Display MSE"] = np.maximum(_plot_frame["MSE"], 1e-3)
    _highlight_degree = int(s3_degree.value)
    _color_scale = alt.Scale(domain=["Training", "Unseen data"], range=[train_color, test_color])

    _lines = alt.Chart(_plot_frame).mark_line(point=True, strokeWidth=2.6).encode(
        x=alt.X("Degree:Q", scale=alt.Scale(domain=[0, 10]), axis=alt.Axis(tickMinStep=1), title="Polynomial degree"),
        y=alt.Y("Display MSE:Q", title="MSE", scale=alt.Scale(type="log")),
        color=alt.Color("Dataset:N", scale=_color_scale),
        tooltip=[
            alt.Tooltip("Degree:Q", format=".0f"),
            alt.Tooltip("Dataset:N"),
            alt.Tooltip("MSE:Q", format=".2f"),
        ],
    )
    _marker = alt.Chart(pd.DataFrame({"Degree": [_highlight_degree]})).mark_rule(
        color=chosen_color,
        strokeDash=[7, 4],
        strokeWidth=2.0,
    ).encode(x="Degree:Q")
    _chart = finish_chart(_lines + _marker)

    _new_rows = _curve_frame[_curve_frame["Dataset"] == "Unseen data"].reset_index(drop=True)
    _best_new_degree = int(_new_rows.loc[_new_rows["MSE"].idxmin(), "Degree"])
    _sidebar = sidebar(
        widgets=[s3_degree, s3_recreate],
        metrics=[
            ("Highlighted degree", str(_highlight_degree)),
            ("Best on new data", str(_best_new_degree)),
        ],
    )
    _layout = mo.vstack(
        [
            section_md(
                "3. Model Complexity: How Overfitting Shows Up",
                "Training MSE rewards flexibility. MSE on new data reveals when that flexibility stops helping.",
                (
                    "This section compares training MSE with MSE on new, unseen data across polynomial degrees from 0 to 10. As the model gets more flexible, training MSE usually drops, but MSE on new data can start rising again.\n\n"
                    "The y-axis is on a log scale so you can still see the full range, from tiny training errors for high-degree polynomials to the larger unseen-data errors that matter for model choice."
                ),
            ),
            two_col(_sidebar, _chart),
            note_md("When the polynomial starts chasing the sample too closely, training error can keep improving even after generalization has started getting worse."),
            takeaway_md("The classic U-shape belongs to MSE on new, unseen data, not to training MSE."),
            questions_md(
                [
                    "Why does the training curve usually keep falling as degree increases?",
                    "Why does the test curve does not?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s4_controls(counter_button, mo):
    s4_degree = mo.ui.slider(0, 5, value=3, step=1, label="Highlighted degree")
    s4_runs = mo.ui.slider(4, 100, value=8, step=1, label="Number of fitted models")
    s4_recreate = counter_button(label="Recreate data", kind="success")
    return s4_degree, s4_recreate, s4_runs


@app.cell
def s4_reference_frame(evaluate_bias_variance_curves):
    s4_reference_frame = evaluate_bias_variance_curves(
        seed=930,
        n_resamples=700,
        degree_grid=tuple(range(6)),
        sample_size=25,
        noise_std=0.58,
        x_grid_points=320,
    )
    return (s4_reference_frame,)


@app.cell
def s4_section(
    age_domain,
    age_from_unit,
    alt,
    bias_color,
    bias_variance_summary,
    chosen_color,
    finish_chart,
    mo,
    mse_color,
    note_md,
    np,
    pd,
    plot_domain,
    questions_md,
    s4_degree,
    s4_recreate,
    s4_reference_frame,
    s4_runs,
    section_md,
    sidebar,
    takeaway_md,
    truth_color,
    two_col,
    variance_color,
    y_domain,
):
    _seed = 300 + int(s4_recreate.value or 0)
    _degree = int(s4_degree.value)
    _summary = bias_variance_summary(
        degree=_degree,
        seed=_seed,
        n_resamples=int(s4_runs.value),
        sample_size=11,
    )
    _curve_frame = s4_reference_frame.copy()
    _curve_frame["Display Value"] = np.minimum(_curve_frame["Value"], 3.0)

    _truth_frame = pd.DataFrame(
        {
            "Age": age_from_unit(_summary["x_grid"]),
            "Value": _summary["truth"],
        }
    )
    _average_frame = pd.DataFrame(
        {
            "Age": age_from_unit(_summary["x_grid"]),
            "Value": np.clip(_summary["average"], plot_domain[0], plot_domain[1]),
        }
    )
    _fit_rows = []
    for _index, _prediction in enumerate(_summary["plot_predictions"]):
        _fit_rows.append(
            pd.DataFrame(
                {
                    "Age": age_from_unit(_summary["x_grid"]),
                    "Value": _prediction,
                    "Model": f"Fit {_index + 1}",
                }
            )
        )
    _fit_frame = pd.concat(_fit_rows, ignore_index=True)

    _fits = alt.Chart(_fit_frame).mark_line(color="#A7B3BF", opacity=0.38, strokeWidth=1.4, clip=True).encode(
        x=alt.X("Age:Q", scale=alt.Scale(domain=list(age_domain)), title="Age"),
        y=alt.Y("Value:Q", scale=alt.Scale(domain=list(y_domain)), title="Life satisfaction"),
        detail="Model:N",
    )
    _truth = alt.Chart(_truth_frame).mark_line(color=truth_color, strokeDash=[6, 4], strokeWidth=2.2, clip=True).encode(
        x="Age:Q",
        y="Value:Q",
    )
    _average = alt.Chart(_average_frame).mark_line(color=chosen_color, strokeWidth=3.0, clip=True).encode(
        x="Age:Q",
        y="Value:Q",
    )
    _left_chart = finish_chart(_fits + _truth + _average, width=320, height=250)

    _color_scale = alt.Scale(
        domain=["E[MSE]", "Bias²", "Variance"],
        range=[mse_color, bias_color, variance_color],
    )
    _right_chart = finish_chart(
        alt.Chart(_curve_frame)
        .mark_line(point=True, strokeWidth=2.5)
        .encode(
            x=alt.X("Degree:Q", scale=alt.Scale(domain=[0, 5]), axis=alt.Axis(tickMinStep=1), title="Polynomial degree"),
            y=alt.Y("Display Value:Q", title="Error component", scale=alt.Scale(domain=[0.0, 3.0])),
            color=alt.Color("Metric:N", scale=_color_scale),
            tooltip=[
                alt.Tooltip("Degree:Q", format=".0f"),
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Value:Q", format=".3f"),
            ],
        ),
        width=360,
        height=250,
    )
    _plots = mo.hstack([_left_chart, _right_chart], widths=[0.95, 1.08], gap=0.24, wrap=True, align="start")

    _mse_rows = _curve_frame[_curve_frame["Metric"] == "E[MSE]"].reset_index(drop=True)
    _best_degree = int(_mse_rows.loc[_mse_rows["Value"].idxmin(), "Degree"])
    _sidebar = sidebar(
        widgets=[s4_degree, s4_runs, s4_recreate],
        metrics=[
            ("Highlighted degree", str(_degree)),
            ("Models shown", str(int(s4_runs.value))),
            ("Bias²", f"{float(_summary['bias_sq']):.3f}"),
            ("Variance", f"{float(_summary['variance']):.3f}"),
            ("E[MSE]", f"{float(_summary['expected_mse']):.3f}"),
            ("Lowest-E[MSE] degree", str(_best_degree)),
        ],
    )
    _layout = mo.vstack(
        [
            section_md(
                "4. The Bias-Variance Trade-Off",
                "Imagine fitting the same model class again and again on fresh random samples.",
                (
                    "The left plot shows several fits of the same degree on different samples. The dashed black line is the true relationship, and the green line is the average prediction across fitted models.\n\n"
                    "Bias is the error that comes from the model being systematically too simple or too rigid: on average, its predictions miss the true pattern. Variance is the error that comes from instability: if we collect a different sample, the fitted curve can move a lot.\n\n"
                    "The right plot summarizes the long-run trade-off across degrees 0 to 5. Here bias² is computed from the best population polynomial of each degree, while variance is estimated from repeated noisy samples, so the decomposition reflects the theoretical model class rather than the small sample shown on the left."
                ),
            ),
            mo.md(r"$\mathbb{E}[\mathrm{MSE}] = \mathbb{E}\!\left[(Y - \hat f(X))^2\right] = \mathrm{Bias}^2 + \mathrm{Variance} + \sigma^2$"),
            mo.md(r"$\mathrm{Bias}^2 = \mathbb{E}_X\!\left[(\mathbb{E}[\hat f(X)] - f(X))^2\right], \qquad \mathrm{Variance} = \mathbb{E}_X\!\left[\mathrm{Var}(\hat f(X))\right]$"),
            mo.md("Bias asks: if we averaged over many possible training sets, how far would the model still be from the truth? Variance asks: how much would the fitted model change from one training set to another?"),
            two_col(_sidebar, _plots),
            note_md("Low-degree models usually pay in bias. More flexible models usually pay in variance. The right-hand plot is restricted to degrees 0 to 5 and clipped at 3 so the trade-off stays readable."),
            takeaway_md("Bias and variance pull in opposite directions, so the best model complexity is usually a compromise."),
            questions_md(
                [
                    "Which part of the curve is dominated more by bias, and which part is dominated more by variance?",
                    "Why can the degree with the lowest MSE sit between the lowest-bias and lowest-variance degrees?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s5_controls(counter_button, mo):
    s5_n = mo.ui.slider(40, 220, value=80, step=10, label="Development sample size N")
    s5_recreate = counter_button(label="Recreate data", kind="success")
    return s5_n, s5_recreate


@app.cell
def s5_section(
    alt,
    build_selection_data,
    chosen_color,
    degrees,
    evaluate_validation_curves,
    finish_chart,
    mo,
    note_md,
    np,
    pd,
    questions_md,
    s5_n,
    s5_recreate,
    section_md,
    sidebar,
    takeaway_md,
    train_color,
    two_col,
    valid_color,
):
    _seed = 410 + int(s5_recreate.value or 0)
    _n = int(s5_n.value)
    _x_dev, _y_dev, _, _ = build_selection_data(seed=_seed, development_n=_n)
    _frame, _summary = evaluate_validation_curves(
        x_dev=_x_dev,
        y_dev=_y_dev,
        degree_grid=degrees,
        split_seed=_seed + 5,
        train_frac=0.70,
    )
    _plot_frame = _frame.copy()
    _plot_frame["Display MSE"] = np.minimum(_plot_frame["MSE"], 1.2)
    _color_scale = alt.Scale(domain=["Training", "Validation"], range=[train_color, valid_color])
    _chart = alt.Chart(_plot_frame).mark_line(point=True, strokeWidth=2.6).encode(
        x=alt.X("Degree:Q", scale=alt.Scale(domain=[0, 10]), axis=alt.Axis(tickMinStep=1), title="Polynomial degree"),
        y=alt.Y("Display MSE:Q", title="MSE", scale=alt.Scale(domain=[0.0, 1.2])),
        color=alt.Color("Dataset:N", scale=_color_scale),
        tooltip=[alt.Tooltip("Degree:Q", format=".0f"), alt.Tooltip("Dataset:N"), alt.Tooltip("MSE:Q", format=".2f")],
    )
    _rule = alt.Chart(pd.DataFrame({"Degree": [_summary["chosen_degree"]]})).mark_rule(
        color=chosen_color,
        strokeDash=[7, 4],
        strokeWidth=2.0,
    ).encode(x="Degree:Q")
    _chart = finish_chart(_chart + _rule, height=250)

    _sidebar = sidebar(
        widgets=[s5_n, s5_recreate],
        metrics=[
            ("Validation chose", f"Degree {int(_summary['chosen_degree'])}"),
            ("Training / validation", f"{int(_summary['train_count'])} / {int(_summary['validation_count'])}"),
        ],
    )
    _layout = mo.vstack(
        [
            section_md(
                "5. Picking Model Complexity with Validation Data",
                "Now we compare ten candidate degrees using a 70:30 training-validation split inside the development sample.",
                (
                    "We often want to compare several models before evaluating the performance of the best one.\n\n"
                    "Different degrees can tell very different stories: some underfit, some overfit, and some estimate generalization error much better than others. We need a principled way to choose among them.\n\n"
                    "Here, `N` is the size of the development sample available for model selection. That development sample is split 70:30 into training and validation.\n\n"
                    "The validation curve is the unseen-data proxy used for choosing among the ten candidate degrees. The test set still stays out of sight."
                ),
            ),
            two_col(_sidebar, _chart),
            note_md("Validation is for choosing among models. It is already giving you unseen-data feedback, so the test set does not need to appear yet."),
            takeaway_md("Validation helps you choose model complexity without spending the test set too early."),
            questions_md(
                [
                    "Why is the validation curve more useful for model choice than the training curve?",
                    "How does changing N affect how noisy the validation decision feels?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s6_controls(counter_button, mo):
    s6_degree = mo.ui.slider(0, 10, value=8, step=1, label="Degree chosen for final check")
    s6_n = mo.ui.slider(40, 220, value=80, step=10, label="Development sample size N")
    s6_recreate = counter_button(label="Recreate data", kind="success")
    s6_reveal = mo.ui.switch(value=False, label="Reveal test error")
    return s6_degree, s6_n, s6_recreate, s6_reveal


@app.cell
def s6_section(
    alt,
    build_selection_data,
    degrees,
    evaluate_validation_curves,
    finish_chart,
    fit_polynomial,
    mo,
    mse,
    note_md,
    np,
    pd,
    predict_polynomial,
    questions_md,
    s6_degree,
    s6_n,
    s6_recreate,
    s6_reveal,
    section_md,
    sidebar,
    split_indices,
    takeaway_md,
    train_color,
    two_col,
    valid_color,
):
    _seed = 510 + int(s6_recreate.value or 0)
    _n = int(s6_n.value)
    _chosen_degree = int(s6_degree.value)
    _x_dev, _y_dev, _x_test, _y_test = build_selection_data(seed=_seed, development_n=_n)
    _frame, _summary = evaluate_validation_curves(
        x_dev=_x_dev,
        y_dev=_y_dev,
        degree_grid=degrees,
        split_seed=_seed + 9,
        train_frac=0.70,
    )
    _plot_frame = _frame.copy()
    _plot_frame["Display MSE"] = np.minimum(_plot_frame["MSE"], 1.2)
    _chart = alt.Chart(_plot_frame).mark_line(point=True, strokeWidth=2.6).encode(
        x=alt.X("Degree:Q", scale=alt.Scale(domain=[0, 10]), axis=alt.Axis(tickMinStep=1), title="Polynomial degree"),
        y=alt.Y("Display MSE:Q", title="MSE", scale=alt.Scale(domain=[0.0, 1.2])),
        color=alt.Color("Dataset:N", scale=alt.Scale(domain=["Training", "Validation"], range=[train_color, valid_color])),
        tooltip=[alt.Tooltip("Degree:Q", format=".0f"), alt.Tooltip("Dataset:N"), alt.Tooltip("MSE:Q", format=".2f")],
    )
    _rules = pd.DataFrame(
        {
            "Degree": [int(_summary["chosen_degree"]), _chosen_degree],
            "Label": ["Validation winner", "Chosen for final check"],
        }
    )
    _rules_chart = alt.Chart(_rules).mark_rule(strokeWidth=1.8, strokeDash=[7, 4]).encode(
        x="Degree:Q",
        color=alt.Color(
            "Label:N",
            scale=alt.Scale(domain=["Validation winner", "Chosen for final check"], range=[valid_color, "#B23A48"]),
        ),
    )
    _chart = finish_chart(_chart + _rules_chart, height=250)

    _metrics = [
        ("Validation chose", f"Degree {int(_summary['chosen_degree'])}"),
        ("Chosen for final check", f"Degree {_chosen_degree}"),
        ("Training / validation", f"{int(_summary['train_count'])} / {int(_summary['validation_count'])}"),
        ("Held-out test size", str(len(_x_test))),
    ]
    if bool(s6_reveal.value):
        _train_idx, _, _ = split_indices(len(_x_dev), train_frac=0.70, valid_frac=0.30, seed=_seed + 9)
        _model = fit_polynomial(_x_dev[_train_idx], _y_dev[_train_idx], _chosen_degree)
        _metrics.append(("Test MSE", f"{mse(_y_test, predict_polynomial(_model, _x_test)):.2f}"))

    _sidebar = sidebar(
        widgets=[s6_degree, s6_n, s6_recreate, s6_reveal],
        metrics=_metrics,
    )
    _layout = mo.vstack(
        [
            section_md(
                "6. Test Data for the Final Audit",
                "Only one chosen model should touch the test set.",
                (
                    "Section 5 used unseen data to compare models. Once unseen data are used for comparison, they are no longer a clean final test.\n\n"
                    "The repair is a three-way split: training for fitting, validation for comparing models, and test for one final audit. Pick one degree for the final check, then reveal its test error."
                ),
            ),
            two_col(_sidebar, _chart),
            note_md("The test set is for auditing a chosen model, not for comparing all ten models side by side."),
            takeaway_md("If we keep peeking at test performance while tuning, we start overfitting the model-selection process too."),
            questions_md(
                [
                    "Why should only one chosen model go to the test set?",
                    "What is the difference between using validation and using test data?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s7_controls(counter_button, mo):
    s7_n = mo.ui.slider(40, 220, value=80, step=10, label="Development sample size N")
    s7_recreate = counter_button(label="Recreate data", kind="success")
    return s7_n, s7_recreate


@app.cell
def s7_section(
    alt,
    build_selection_data,
    cap_mse_frame,
    degrees,
    evaluate_validation_curves,
    finish_chart,
    mo,
    note_md,
    pd,
    questions_md,
    s7_n,
    s7_recreate,
    section_md,
    sidebar,
    split_a_color,
    split_b_color,
    takeaway_md,
    two_col,
):
    _seed = 610 + int(s7_recreate.value or 0)
    _n = int(s7_n.value)
    _x_dev, _y_dev, _, _ = build_selection_data(seed=_seed, development_n=_n)
    _split_a, _summary_a = evaluate_validation_curves(
        x_dev=_x_dev,
        y_dev=_y_dev,
        degree_grid=degrees,
        split_seed=_seed + 9,
        train_frac=0.70,
    )
    _split_b, _summary_b = evaluate_validation_curves(
        x_dev=_x_dev,
        y_dev=_y_dev,
        degree_grid=degrees,
        split_seed=_seed + 41,
        train_frac=0.70,
    )
    _plot_frame = pd.concat(
        [
            _split_a[_split_a["Dataset"] == "Validation"].assign(Split="Validation split A"),
            _split_b[_split_b["Dataset"] == "Validation"].assign(Split="Validation split B"),
        ],
        ignore_index=True,
    )
    _plot_frame = cap_mse_frame(_plot_frame)
    _rules = pd.DataFrame(
        {
            "Degree": [int(_summary_a["chosen_degree"]), int(_summary_b["chosen_degree"])],
            "Split": ["Validation split A", "Validation split B"],
        }
    )
    _chart = alt.Chart(_plot_frame).mark_line(point=True, strokeWidth=2.6).encode(
        x=alt.X("Degree:Q", scale=alt.Scale(domain=[0, 10]), axis=alt.Axis(tickMinStep=1), title="Polynomial degree"),
        y=alt.Y("Display MSE:Q", title="Validation MSE"),
        color=alt.Color(
            "Split:N",
            scale=alt.Scale(domain=["Validation split A", "Validation split B"], range=[split_a_color, split_b_color]),
        ),
        tooltip=[alt.Tooltip("Degree:Q", format=".0f"), alt.Tooltip("Split:N"), alt.Tooltip("MSE:Q", format=".2f")],
    )
    _rules_chart = alt.Chart(_rules).mark_rule(strokeDash=[7, 4], strokeWidth=1.8).encode(
        x="Degree:Q",
        color=alt.Color(
            "Split:N",
            scale=alt.Scale(domain=["Validation split A", "Validation split B"], range=[split_a_color, split_b_color]),
            legend=None,
        ),
    )
    _chart = finish_chart(_chart + _rules_chart, height=250)
    _sidebar = sidebar(
        widgets=[s7_n, s7_recreate],
        metrics=[
            ("Split A chose", f"Degree {int(_summary_a['chosen_degree'])}"),
            ("Split B chose", f"Degree {int(_summary_b['chosen_degree'])}"),
            ("Training / validation", f"{int(_summary_a['train_count'])} / {int(_summary_a['validation_count'])}"),
        ],
    )
    _layout = mo.vstack(
        [
            section_md(
                "7. Split Dependence in Validation",
                "One drawback of the train-validation-test split is that validation MSE can be highly variable.",
                (
                    "This section shows split instability with the same `N` and the same 70:30 rule. Two different train-validation splits can produce two different recommended degrees, even when the model family and the data-generating process are unchanged.\n\n"
                    "What changed here is only the partition of the same development sample."
                ),
            ),
            two_col(_sidebar, _chart),
            note_md("If two reasonable validation splits disagree, that is evidence that one split is too fragile to carry the whole decision."),
            takeaway_md("When one split feels noisy, the next step is usually to average more information rather than to over-trust one partition."),
            questions_md(
                [
                    "What changed here besides the split itself?",
                    "What would you do next if two validation splits disagree?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s8_controls(counter_button, mo):
    s8_n = mo.ui.slider(40, 220, value=80, step=10, label="Development sample size N")
    s8_recreate = counter_button(label="Recreate data", kind="success")
    return s8_n, s8_recreate


@app.cell
def s8_section(
    alt,
    build_selection_data,
    cap_mse_frame,
    cv_color,
    degrees,
    evaluate_cv_curves,
    evaluate_validation_curves,
    finish_chart,
    fit_polynomial,
    mo,
    mse,
    note_md,
    pd,
    predict_polynomial,
    questions_md,
    s8_n,
    s8_recreate,
    section_md,
    sidebar,
    split_a_color,
    split_b_color,
    takeaway_md,
    two_col,
):
    _seed = 710 + int(s8_recreate.value or 0)
    _n = int(s8_n.value)
    _x_dev, _y_dev, _x_test, _y_test = build_selection_data(seed=_seed, development_n=_n)
    _split_a, _summary_a = evaluate_validation_curves(
        x_dev=_x_dev,
        y_dev=_y_dev,
        degree_grid=degrees,
        split_seed=_seed + 9,
        train_frac=0.70,
    )
    _split_b, _summary_b = evaluate_validation_curves(
        x_dev=_x_dev,
        y_dev=_y_dev,
        degree_grid=degrees,
        split_seed=_seed + 41,
        train_frac=0.70,
    )
    _cv_frame = evaluate_cv_curves(_x_dev, _y_dev, degrees, n_splits=5, seed=_seed + 73)
    _cv_degree = int(_cv_frame.loc[_cv_frame["CV MSE"].idxmin(), "Degree"])
    _final_model = fit_polynomial(_x_dev, _y_dev, _cv_degree)
    _final_test_mse = mse(_y_test, predict_polynomial(_final_model, _x_test))

    _comparison = pd.concat(
        [
            _split_a[_split_a["Dataset"] == "Validation"][["Degree", "MSE"]].assign(Curve="Validation split A"),
            _split_b[_split_b["Dataset"] == "Validation"][["Degree", "MSE"]].assign(Curve="Validation split B"),
            _cv_frame.rename(columns={"CV MSE": "MSE"}).assign(Curve="5-fold CV"),
        ],
        ignore_index=True,
    )
    _plot_frame = cap_mse_frame(_comparison)
    _rules = pd.DataFrame(
        {
            "Degree": [int(_summary_a["chosen_degree"]), int(_summary_b["chosen_degree"]), _cv_degree],
            "Curve": ["Validation split A", "Validation split B", "5-fold CV"],
        }
    )
    _chart = alt.Chart(_plot_frame).mark_line(point=True, strokeWidth=2.6).encode(
        x=alt.X("Degree:Q", scale=alt.Scale(domain=[0, 10]), axis=alt.Axis(tickMinStep=1), title="Polynomial degree"),
        y=alt.Y("Display MSE:Q", title="Selection error"),
        color=alt.Color(
            "Curve:N",
            scale=alt.Scale(
                domain=["Validation split A", "Validation split B", "5-fold CV"],
                range=[split_a_color, split_b_color, cv_color],
            ),
        ),
        tooltip=[alt.Tooltip("Degree:Q", format=".0f"), alt.Tooltip("Curve:N"), alt.Tooltip("MSE:Q", format=".2f")],
    )
    _rules_chart = alt.Chart(_rules).mark_rule(strokeDash=[7, 4], strokeWidth=1.8).encode(
        x="Degree:Q",
        color=alt.Color(
            "Curve:N",
            scale=alt.Scale(
                domain=["Validation split A", "Validation split B", "5-fold CV"],
                range=[split_a_color, split_b_color, cv_color],
            ),
            legend=None,
        ),
    )
    _chart = finish_chart(_chart + _rules_chart, height=250)
    _sidebar = sidebar(
        widgets=[s8_n, s8_recreate],
        metrics=[
            ("Split A chose", f"Degree {int(_summary_a['chosen_degree'])}"),
            ("Split B chose", f"Degree {int(_summary_b['chosen_degree'])}"),
            ("5-fold CV chose", f"Degree {_cv_degree}"),
            ("Outer test after CV", f"{_final_test_mse:.2f}"),
        ],
    )
    _layout = mo.vstack(
        [
            section_md(
                "8. Cross-Validation for Stability",
                "Cross-validation uses the same development sample more efficiently than one fixed 70:30 split.",
                (
                    "Cross-validation is the remedy for instability in model comparison. Instead of betting everything on one validation split, we average validation performance across several folds and then choose the degree with the lowest average error.\n\n"
                    "Cross-validation is not a replacement for the final test set. After choosing the degree, we retrain on all development data and use the test set once for a final check."
                ),
            ),
            two_col(_sidebar, _chart),
            note_md("Cross-validation does not make noise disappear, but it usually reduces the chance that one arbitrary split dominates the complexity decision."),
            takeaway_md("Use validation or cross-validation to choose complexity, retrain on all development data, and save the test set for one final check."),
            questions_md(
                [
                    "Why is cross-validation usually less sensitive than one validation split?",
                    "After choosing the degree with CV, why do we still keep a test set?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


if __name__ == "__main__":
    app.run()
