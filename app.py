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
    validation_winner_color = "#7B4EA3"
    final_check_color = "#007C89"
    holdout_color = "#C9A227"
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
        final_check_color,
        holdout_color,
        mse_color,
        residual_color,
        split_a_color,
        split_b_color,
        style,
        test_color,
        train_color,
        truth_color,
        valid_color,
        validation_winner_color,
        variance_color,
    )


@app.cell
def lecture_config():
    age_domain = (18.0, 75.0)
    y_domain = (4.0, 8.5)
    plot_domain = (3.95, 8.55)
    degrees = tuple(range(0, 11))
    selection_degrees = tuple(range(0, 11))
    return age_domain, degrees, plot_domain, selection_degrees, y_domain


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

    return (
        build_selection_data,
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
            "variance": float(np.mean(np.var(prediction_array, axis=0, ddof=1))),
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
        def run_bv(param_vals, n_resamples: int, n_train: int, fit_fn) -> pd.DataFrame:
            rows: list[dict[str, float | int]] = []
            for param_value in param_vals:
                prediction_matrix = np.column_stack(
                    [fit_fn(run_index, param_value, n_train) for run_index in range(n_resamples)]
                )
                rows.append(
                    {
                        "param": int(param_value),
                        "bias2": float(np.mean((prediction_matrix.mean(axis=1) - truth) ** 2)),
                        "variance": float(np.mean(np.var(prediction_matrix, axis=1, ddof=1))),
                        "test_mse": float(np.mean((prediction_matrix - truth[:, None]) ** 2)),
                    }
                )
            return pd.DataFrame(rows)

        def fit_fn(run_index: int, degree: int, n_train: int) -> np.ndarray:
            x_sample, y_sample = generate_points(
                n_train,
                noise_std=noise_std,
                seed=seed + degree * 100_000 + 13 * run_index,
            )
            model = fit_polynomial(x_sample, y_sample, degree)
            return predict_polynomial(model, x_grid)

        rows: list[dict[str, float | int | str]] = []
        x_grid = np.linspace(0.0, 1.0, x_grid_points)
        truth = true_function(x_grid)
        decomposition = run_bv(degree_grid, n_resamples, sample_size, fit_fn)
        for component in decomposition.itertuples(index=False):
            rows.extend(
                [
                    {"Degree": component.param, "Metric": "E[MSE]", "Value": float(component.test_mse + noise_std**2)},
                    {"Degree": component.param, "Metric": "Bias²", "Value": float(component.bias2)},
                    {"Degree": component.param, "Metric": "Variance", "Value": float(component.variance)},
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
                "**Vocabulary.** We use **unseen-data MSE** for performance on data not used to fit the displayed model. We reserve **held-out test MSE** for the final one-time estimate after one model has been chosen."
            ),
            mo.md(
                "How to use it: move the sliders to change the polynomial model, press the `Recreate` buttons to draw another random sample, and use the reveal switches only after making a prediction yourself. Each section is independent, so you can explore one idea without losing your place in the rest of the app."
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
                    "Before we compare models, we need a metric for how far the predictions are from the observed outcomes. In this section that metric is **mean squared error (MSE)**. \n"
                    r"$\mathrm{MSE} = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat y_i)^2$"
                    "\n\n"
                    "Use the polynomial-degree slider to change the fitted relationship between age and life satisfaction, and use `Recreate data` to draw another sample from the same population."
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
                    "What is the difference between a residual and MSE?",
                    "Why is lower training MSE not enough to choose a predictive model?",
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
    s2_reveal_test = mo.ui.switch(value=False, label="Reveal unseen data")
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
        _metrics.append(("Unseen-data MSE", f"{mse(_y_test, predict_polynomial(_model, _x_test)):.2f}"))

    _chart = finish_chart(alt.layer(*_layers))
    _sidebar = sidebar(
        widgets=[s2_degree, s2_recreate, s2_reveal_test, s2_show_train],
        metrics=_metrics,
    )
    _layout = mo.vstack(
        [
            section_md(
                "2. From Training Data to Unseen Data",
                "To estimate generalization error, training data is not enough. We need unseen data.",
                (
                    "**Unseen data** means data not used to fit the model we are evaluating. It can be a validation set, a held-out test set, or another sample from the same population.\n\n"
                    "Here the line is fitted using the training data only. Then the switch reveals unseen points from the same population without changing the fitted line. The resulting **unseen-data MSE** is a first estimate of how well the model generalizes."
                ),
            ),
            two_col(_sidebar, _chart),
            note_md("The fitted line does not move when the unseen points appear. The unseen points are only for evaluation, not for fitting."),
            takeaway_md("Training MSE answers how well the model fits data it already saw. Unseen-data MSE asks whether that fit generalizes."),
            questions_md(
                [
                    "What is unseen-data MSE, and how is it different from training MSE?",
                    "Why should the fitted line stay fixed when unseen points are revealed?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s3_controls(counter_button, mo):
    s3_degree = mo.ui.slider(0, 10, value=8, step=1, label="Highlighted polynomial degree")
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
    _highlight_rows = _curve_frame[_curve_frame["Degree"] == _highlight_degree]
    _highlight_train_mse = float(_highlight_rows.loc[_highlight_rows["Dataset"] == "Training", "MSE"].iloc[0])
    _highlight_test_mse = float(_highlight_rows.loc[_highlight_rows["Dataset"] == "Unseen data", "MSE"].iloc[0])
    _sidebar = sidebar(
        widgets=[s3_degree, s3_recreate],
        metrics=[
            ("Flexibility", f"Degree {_highlight_degree}"),
            ("Highlighted train MSE", f"{_highlight_train_mse:.2f}"),
            ("Highlighted unseen-data MSE", f"{_highlight_test_mse:.2f}"),
            ("Best unseen-data degree", str(_best_new_degree)),
        ],
    )
    _layout = mo.vstack(
        [
            section_md(
                "3. Model Complexity: How Overfitting Shows Up",
                "Here, flexibility means polynomial degree. Higher degree means a more flexible curve.",
                (
                    "This section compares training MSE with **unseen-data MSE** across polynomial degrees from 0 to 10. As degree increases, the model has more freedom to bend toward the training sample. Training MSE usually drops, but unseen-data MSE can start rising again.\n\n"
                    "The y-axis is on a log scale so you can still see the full range, from tiny training errors for high-degree polynomials to the larger unseen-data errors that matter for model choice."
                ),
            ),
            two_col(_sidebar, _chart),
            note_md("When the polynomial starts chasing the sample too closely, training error can keep improving even after generalization has started getting worse."),
            takeaway_md("The classic U-shape belongs to unseen-data MSE, not to training MSE."),
            questions_md(
                [
                    "In this section, what exactly does model flexibility mean?",
                    "Why can training MSE improve while unseen-data MSE gets worse?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s4_controls(counter_button, mo):
    s4_degree = mo.ui.slider(0, 5, value=3, step=1, label="Highlighted polynomial degree")
    s4_runs = mo.ui.slider(4, 100, value=8, step=1, label="Models shown in top plot")
    s4_recreate = counter_button(label="Recreate data", kind="success")
    return s4_degree, s4_recreate, s4_runs


@app.cell
def s4_reference_frame(evaluate_bias_variance_curves):
    s4_reference_frame = evaluate_bias_variance_curves(
        seed=930,
        n_resamples=1000,
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
    _left_chart = finish_chart(
        (_fits + _truth + _average).properties(title="Repeated fits at the highlighted degree"),
        width=560,
        height=220,
    )

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
            color=alt.Color("Metric:N", scale=_color_scale, legend=alt.Legend(title="Component")),
            tooltip=[
                alt.Tooltip("Degree:Q", format=".0f"),
                alt.Tooltip("Metric:N"),
                alt.Tooltip("Value:Q", format=".3f"),
            ],
        ),
        width=560,
        height=250,
    )
    _right_chart = _right_chart.properties(title="Bias-variance decomposition from 1,000 fitted models per degree")
    _plots = mo.vstack([_left_chart, _right_chart], gap=0.12)

    _mse_rows = _curve_frame[_curve_frame["Metric"] == "E[MSE]"].reset_index(drop=True)
    _best_degree = int(_mse_rows.loc[_mse_rows["Value"].idxmin(), "Degree"])
    _sidebar = sidebar(
        widgets=[s4_degree, s4_runs, s4_recreate],
        metrics=[
            ("Flexibility", f"Degree {_degree}"),
            ("Top-plot models", str(int(s4_runs.value))),
            ("Decomposition models", "1000 per degree"),
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
                "In this app, flexibility means polynomial degree.",
                (
                    "First focus on **E[MSE]**: the prediction error we would expect after repeatedly drawing a training sample, fitting a degree-p polynomial, and evaluating it on unseen data.\n\n"
                    "Then split that expected error into **bias** and **variance**. Bias is systematic error: if we averaged predictions over many possible training samples, how far would that average still be from the true relationship? Variance is instability: how much do predictions change from one training sample to another?\n\n"
                    "The top plot shows several fits of the highlighted degree on different samples. The dashed black line is the true relationship, and the green line is the average prediction across fitted models. The bottom plot uses 1,000 fitted models for every degree from 0 to 5."
                ),
            ),
            mo.md(r"$\mathbb{E}[\mathrm{MSE}] = \mathbb{E}\!\left[(Y - \hat f(X))^2\right] = \mathrm{Bias}^2 + \mathrm{Variance} + \sigma^2$"),
            mo.md(r"$\mathrm{Bias}^2 = \mathbb{E}_X\!\left[(\mathbb{E}_{D}[\hat f_D(X)] - f(X))^2\right]$"),
            mo.md(r"$\mathrm{Variance} = \mathbb{E}_X\!\left[\mathbb{E}_{D}\!\left[(\hat f_D(X) - \mathbb{E}_{D}[\hat f_D(X)])^2\right]\right]$"),
            two_col(_sidebar, _plots),
            note_md("Low-degree models usually pay in bias. Higher-degree, more flexible models usually pay in variance. The decomposition plot is restricted to degrees 0 to 5 and clipped at 3 so the trade-off stays readable."),
            takeaway_md("Bias and variance pull in opposite directions, so the best model complexity is usually a compromise."),
            questions_md(
                [
                    "How would you explain bias as systematic error and variance as instability?",
                    "Why can the degree with the lowest E[MSE] be a compromise between bias and variance?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s5_controls(counter_button, mo):
    s5_n = mo.ui.slider(10, 220, value=10, step=10, label="Development sample size N")
    s5_recreate = counter_button(label="Recreate data", kind="success")
    return s5_n, s5_recreate


@app.cell
def s5_section(
    alt,
    build_selection_data,
    chosen_color,
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
    selection_degrees,
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
        degree_grid=selection_degrees,
        split_seed=_seed + 5,
        train_frac=0.50,
    )
    _plot_frame = _frame.copy()
    _plot_frame["Display MSE"] = np.maximum(_plot_frame["MSE"], 1e-3)
    _color_scale = alt.Scale(domain=["Training", "Validation"], range=[train_color, valid_color])
    _chart = alt.Chart(_plot_frame).mark_line(point=True, strokeWidth=2.6).encode(
        x=alt.X("Degree:Q", scale=alt.Scale(domain=[0, 10]), axis=alt.Axis(tickMinStep=1), title="Polynomial degree"),
        y=alt.Y("Display MSE:Q", title="MSE (log scale)", scale=alt.Scale(type="log")),
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
                "Now we compare eleven candidate degrees using a 50:50 training-validation split inside the development sample.",
                (
                    "We often want to compare several models before estimating the performance of the final chosen model.\n\n"
                    "Different degrees can tell very different stories: some underfit, some overfit, and some have much lower unseen-data MSE than others. We need a principled way to choose among them.\n\n"
                    "Here, `N` is the size of the development sample available for model selection. That development sample is split 50:50 into training and validation.\n\n"
                    "The validation curve is an **unseen-data MSE** curve used for choosing among degrees 0 through 10. It is not the final held-out test MSE. The held-out test set stays out of sight until one model has been chosen."
                ),
            ),
            two_col(_sidebar, _chart),
            note_md("Validation data are unseen by the fitted model, but they are still used during model selection. That is why validation MSE and held-out test MSE have different jobs."),
            takeaway_md("Validation helps you choose model complexity without spending the held-out test set too early."),
            questions_md(
                [
                    "Why is validation MSE an unseen-data MSE used for model choice?",
                    "If you recreate the development sample, how stable is the validation-chosen degree?",
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
    s6_n = mo.ui.slider(10, 220, value=10, step=10, label="Development sample size N")
    s6_recreate = counter_button(label="Recreate data", kind="success")
    s6_reveal = mo.ui.switch(value=False, label="Reveal held-out test error")
    return s6_degree, s6_n, s6_recreate, s6_reveal


@app.cell
def s6_section(
    alt,
    build_selection_data,
    evaluate_validation_curves,
    final_check_color,
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
    selection_degrees,
    sidebar,
    takeaway_md,
    train_color,
    two_col,
    valid_color,
    validation_winner_color,
):
    _seed = 510 + int(s6_recreate.value or 0)
    _n = int(s6_n.value)
    _chosen_degree = int(s6_degree.value)
    _x_dev, _y_dev, _x_test, _y_test = build_selection_data(seed=_seed, development_n=_n)
    _frame, _summary = evaluate_validation_curves(
        x_dev=_x_dev,
        y_dev=_y_dev,
        degree_grid=selection_degrees,
        split_seed=_seed + 9,
        train_frac=0.50,
    )
    _plot_frame = _frame.copy()
    _plot_frame["Display MSE"] = np.maximum(_plot_frame["MSE"], 1e-3)
    _mse_chart = alt.Chart(_plot_frame).mark_line(point=True, strokeWidth=2.6).encode(
        x=alt.X("Degree:Q", scale=alt.Scale(domain=[0, 10]), axis=alt.Axis(tickMinStep=1), title="Polynomial degree"),
        y=alt.Y("Display MSE:Q", title="MSE (log scale)", scale=alt.Scale(type="log")),
        color=alt.Color(
            "Dataset:N",
            scale=alt.Scale(domain=["Training", "Validation"], range=[train_color, valid_color]),
            legend=alt.Legend(title="MSE curve"),
        ),
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
            scale=alt.Scale(
                domain=["Validation winner", "Chosen for final check"],
                range=[validation_winner_color, final_check_color],
            ),
            legend=alt.Legend(title="Reference line"),
        ),
    )
    _chart = finish_chart(alt.layer(_mse_chart, _rules_chart).resolve_scale(color="independent"), height=250)

    _chosen_validation_mse = float(
        _frame.loc[
            (_frame["Dataset"] == "Validation") & (_frame["Degree"] == _chosen_degree),
            "MSE",
        ].iloc[0]
    )
    _metrics = [
        ("Validation chose", f"Degree {int(_summary['chosen_degree'])}"),
        ("Chosen for final check", f"Degree {_chosen_degree}"),
        ("Chosen degree validation MSE", f"{_chosen_validation_mse:.2f}"),
        ("Training / validation", f"{int(_summary['train_count'])} / {int(_summary['validation_count'])}"),
        ("Held-out test size", str(len(_x_test))),
    ]
    if bool(s6_reveal.value):
        _model = fit_polynomial(_x_dev, _y_dev, _chosen_degree)
        _metrics.append(("Held-out test MSE", f"{mse(_y_test, predict_polynomial(_model, _x_test)):.2f}"))

    _sidebar = sidebar(
        widgets=[s6_degree, s6_n, s6_recreate, s6_reveal],
        metrics=_metrics,
    )
    _layout = mo.vstack(
        [
            section_md(
                "6. Held-Out Test Data for the Final Audit",
                "Only one chosen model should touch the held-out test set.",
                (
                    "Section 5 used validation data to compare models. Validation data are unseen by each fitted model, but they are not a clean final audit because they influenced the choice of degree.\n\n"
                    "The repair is a three-way split: training for fitting, validation for comparing degrees 0 through 10, and a **held-out test set** for one final audit. Pick one degree for the final check, retrain that one model on the full development sample, then reveal its held-out test MSE once."
                ),
            ),
            two_col(_sidebar, _chart),
            note_md("The held-out test set is for auditing one chosen model, not for comparing all candidate degrees side by side."),
            takeaway_md("If we keep peeking at held-out test performance while tuning, we start overfitting the model-selection process too."),
            questions_md(
                [
                    "Why is the held-out test set used only after the degree has been chosen?",
                    "What would go wrong if we used held-out test MSE to choose among degrees?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s6b_controls(counter_button, mo):
    s6b_n = mo.ui.slider(4, 24, value=8, step=1, label="Validation sample size N")
    s6b_recreate = counter_button(label="Recreate data", kind="success")
    s6b_reveal = mo.ui.switch(value=False, label="Reveal held-out test data")
    return s6b_n, s6b_recreate, s6b_reveal


@app.cell
def s6b_section(
    alt,
    finish_chart,
    holdout_color,
    mo,
    mse,
    note_md,
    np,
    pd,
    questions_md,
    s6b_n,
    s6b_recreate,
    s6b_reveal,
    section_md,
    sidebar,
    split_a_color,
    split_b_color,
    takeaway_md,
    two_col,
    valid_color,
):
    _seed = 560 + int(s6b_recreate.value or 0)
    _n = int(s6b_n.value)
    _rng = np.random.default_rng(_seed)
    _x_valid = np.arange(1, _n + 1)
    _y_valid = _rng.normal(0.0, 1.0, _n)
    _test_n = 40
    _x_test = np.arange(_n + 3, _n + 3 + _test_n)
    _y_test = _rng.normal(0.0, 1.0, _test_n)
    _x_line_max = (_x_test[-1] + 0.5) if bool(s6b_reveal.value) else (_n + 0.5)

    _pred_plus = np.ones(_n)
    _pred_minus = -np.ones(_n)
    _valid_mse_plus = mse(_y_valid, _pred_plus)
    _valid_mse_minus = mse(_y_valid, _pred_minus)
    _chosen_value = 1.0 if _valid_mse_plus <= _valid_mse_minus else -1.0
    _chosen_label = "Model A: y = 1" if _chosen_value > 0 else "Model B: y = -1"
    _chosen_validation_mse = min(_valid_mse_plus, _valid_mse_minus)
    _test_mse = mse(_y_test, np.full_like(_y_test, _chosen_value))

    _valid_frame = pd.DataFrame({"x": _x_valid, "y": _y_valid})
    _line_frame = pd.DataFrame(
        [
            {"x": 0.5, "y": 1.0, "Model": "Model A: y = 1"},
            {"x": _x_line_max, "y": 1.0, "Model": "Model A: y = 1"},
            {"x": 0.5, "y": -1.0, "Model": "Model B: y = -1"},
            {"x": _x_line_max, "y": -1.0, "Model": "Model B: y = -1"},
        ]
    )
    _model_scale = alt.Scale(
        domain=["Model A: y = 1", "Model B: y = -1"],
        range=[split_a_color, split_b_color],
    )
    _points = alt.Chart(_valid_frame).mark_circle(color=valid_color, size=78, opacity=0.82).encode(
        x=alt.X("x:Q", title="Observation index", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("y:Q", title="Observed y", scale=alt.Scale(domain=[-3.4, 3.4])),
        tooltip=[alt.Tooltip("x:Q", format=".0f"), alt.Tooltip("y:Q", format=".2f")],
    )
    _lines = alt.Chart(_line_frame).mark_line(strokeWidth=2.8).encode(
        x=alt.X("x:Q"),
        y=alt.Y("y:Q"),
        color=alt.Color("Model:N", scale=_model_scale, legend=alt.Legend(title="Candidate model")),
    )
    _zero = alt.Chart(pd.DataFrame({"y": [0.0]})).mark_rule(color="#7A828A", strokeDash=[4, 4]).encode(y="y:Q")
    _scatter_layers = [_points, _lines, _zero]
    if bool(s6b_reveal.value):
        _test_frame = pd.DataFrame({"x": _x_test, "y": _y_test})
        _test_points = alt.Chart(_test_frame).mark_point(
            shape="diamond",
            filled=True,
            color=holdout_color,
            size=62,
            opacity=0.74,
        ).encode(
            x=alt.X("x:Q"),
            y=alt.Y("y:Q"),
            tooltip=[
                alt.Tooltip("x:Q", format=".0f", title="Test observation"),
                alt.Tooltip("y:Q", format=".2f"),
            ],
        )
        _test_separator = alt.Chart(pd.DataFrame({"x": [_n + 1.5]})).mark_rule(
            color=holdout_color,
            strokeDash=[5, 4],
            strokeWidth=1.6,
        ).encode(x="x:Q")
        _test_label = alt.Chart(pd.DataFrame({"x": [_x_test[0]], "y": [3.05], "Label": ["Held-out test"]})).mark_text(
            align="left",
            color=holdout_color,
            fontSize=12,
            fontWeight="bold",
        ).encode(x="x:Q", y="y:Q", text="Label:N")
        _scatter_layers.extend([_test_separator, _test_points, _test_label])
    _scatter_chart = finish_chart(
        alt.layer(*_scatter_layers).properties(title="Small validation sample and two equally bad models"),
        width=410,
        height=245,
    )

    _distribution_rows = []
    for run_index in range(700):
        rng = np.random.default_rng(_seed + 10_000 + run_index)
        y_valid = rng.normal(0.0, 1.0, _n)
        y_test = rng.normal(0.0, 1.0, 120)
        valid_plus = mse(y_valid, np.ones(_n))
        valid_minus = mse(y_valid, -np.ones(_n))
        chosen_value = 1.0 if valid_plus <= valid_minus else -1.0
        _distribution_rows.append(
            {
                "Estimate": "Chosen validation MSE",
                "MSE": min(valid_plus, valid_minus),
            }
        )
        _distribution_rows.append(
            {
                "Estimate": "Held-out test MSE",
                "MSE": mse(y_test, np.full_like(y_test, chosen_value)),
            }
        )
    _distribution_frame = pd.DataFrame(_distribution_rows)
    _mean_frame = _distribution_frame.groupby("Estimate", as_index=False)["MSE"].mean()
    _estimate_scale = alt.Scale(
        domain=["Chosen validation MSE", "Held-out test MSE"],
        range=[valid_color, holdout_color],
    )
    _density_chart = alt.Chart(_distribution_frame).transform_density(
        "MSE",
        groupby=["Estimate"],
        as_=["MSE", "Density"],
        extent=[0.0, 5.6],
    ).mark_area(opacity=0.34).encode(
        x=alt.X("MSE:Q", title="MSE across repeated samples"),
        y=alt.Y("Density:Q", title="Density"),
        color=alt.Color("Estimate:N", scale=_estimate_scale, legend=alt.Legend(title="Estimate")),
    )
    _mean_rules = alt.Chart(_mean_frame).mark_rule(strokeDash=[6, 4], strokeWidth=2.0).encode(
        x=alt.X("MSE:Q"),
        color=alt.Color("Estimate:N", scale=_estimate_scale, legend=None),
        tooltip=[alt.Tooltip("Estimate:N"), alt.Tooltip("MSE:Q", format=".2f", title="Average MSE")],
    )
    _distribution_chart = finish_chart(
        (_density_chart + _mean_rules).properties(title="Selection makes validation error optimistic"),
        width=410,
        height=220,
    )
    _avg_chosen_validation_mse = float(
        _mean_frame.loc[_mean_frame["Estimate"] == "Chosen validation MSE", "MSE"].iloc[0]
    )
    _avg_test_mse = float(_mean_frame.loc[_mean_frame["Estimate"] == "Held-out test MSE", "MSE"].iloc[0])

    _metrics = [
        ("Validation MSE, y = 1", f"{_valid_mse_plus:.2f}"),
        ("Validation MSE, y = -1", f"{_valid_mse_minus:.2f}"),
        ("Validation chooses", _chosen_label),
        ("Chosen validation MSE", f"{_chosen_validation_mse:.2f}"),
        ("Avg chosen validation MSE", f"{_avg_chosen_validation_mse:.2f}"),
        ("Avg held-out test MSE", f"{_avg_test_mse:.2f}"),
    ]
    if bool(s6b_reveal.value):
        _metrics.append(("Held-out test MSE", f"{_test_mse:.2f}"))

    _sidebar = sidebar(
        widgets=[s6b_n, s6b_recreate, s6b_reveal],
        metrics=_metrics,
    )
    _right_panel = mo.vstack([_scatter_chart, _distribution_chart], gap=0.18)
    _layout = mo.vstack(
        [
            section_md(
                "7. Why Validation MSE Is Not the Final Generalization Estimate",
                "After we use validation data to choose a model, the chosen validation MSE is usually too optimistic.",
                (
                    "Here the data are pure noise: `y ~ N(0, 1)`. The two candidate models, `y = 1` and `y = -1`, are equally bad in expectation. With a small validation sample, one model often wins just by luck.\n\n"
                    "The distribution plot repeats that selection process many times. The chosen validation MSE is the smaller of two noisy validation estimates, so it tends to sit below the held-out test MSE of the selected model."
                ),
            ),
            two_col(_sidebar, _right_panel),
            note_md("Validation MSE is useful for choosing among models. Once it has influenced the choice, it is no longer a clean final estimate of generalization error."),
            takeaway_md("Selection makes the winning validation error optimistic; use a held-out test set for the final audit."),
            questions_md(
                [
                    "Why are the two candidate models equally bad before looking at the sample?",
                    "Why is the smaller validation MSE biased downward after model selection?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s7_controls(counter_button, mo):
    s7_n = mo.ui.slider(10, 220, value=10, step=10, label="Development sample size N")
    s7_recreate = counter_button(label="Recreate data", kind="success")
    return s7_n, s7_recreate


@app.cell
def s7_section(
    alt,
    build_selection_data,
    evaluate_validation_curves,
    finish_chart,
    mo,
    note_md,
    np,
    pd,
    questions_md,
    s7_n,
    s7_recreate,
    section_md,
    selection_degrees,
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
        degree_grid=selection_degrees,
        split_seed=_seed + 9,
        train_frac=0.50,
    )
    _split_b, _summary_b = evaluate_validation_curves(
        x_dev=_x_dev,
        y_dev=_y_dev,
        degree_grid=selection_degrees,
        split_seed=_seed + 41,
        train_frac=0.50,
    )
    _plot_frame = pd.concat(
        [
            _split_a[_split_a["Dataset"] == "Validation"].assign(Split="Validation split A"),
            _split_b[_split_b["Dataset"] == "Validation"].assign(Split="Validation split B"),
        ],
        ignore_index=True,
    )
    _plot_frame["Display MSE"] = np.maximum(_plot_frame["MSE"], 1e-3)
    _rules = pd.DataFrame(
        {
            "Degree": [int(_summary_a["chosen_degree"]), int(_summary_b["chosen_degree"])],
            "Split": ["Validation split A", "Validation split B"],
        }
    )
    _chart = alt.Chart(_plot_frame).mark_line(point=True, strokeWidth=2.6).encode(
        x=alt.X("Degree:Q", scale=alt.Scale(domain=[0, 10]), axis=alt.Axis(tickMinStep=1), title="Polynomial degree"),
        y=alt.Y("Display MSE:Q", title="Validation MSE (log scale)", scale=alt.Scale(type="log")),
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
                "8. Split Dependence in Validation",
                "One drawback of the train-validation-test split is that validation MSE can be highly variable.",
                (
                    "This section shows split instability with the same `N` and the same 50:50 rule. Two different train-validation splits can produce two different recommended degrees within the 0-through-10 grid, even when the model family and the data-generating process are unchanged.\n\n"
                    "What changed here is only the partition of the same development sample."
                ),
            ),
            two_col(_sidebar, _chart),
            note_md("If two reasonable validation splits disagree, that is evidence that one split is too fragile to carry the whole decision."),
            takeaway_md("When one split feels noisy, the next step is usually to average more information rather than to over-trust one partition."),
            questions_md(
                [
                    "What changes between validation split A and validation split B?",
                    "Why is split disagreement a reason to average across splits with CV?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


@app.cell
def s8_cv_visual_intro(alt, cv_color, finish_chart, holdout_color, mo, pd, section_md):
    _fold_order = [f"Fold {fold_index}" for fold_index in range(1, 6)] + ["Held-out test"]
    _round_order = [f"Round {round_index}" for round_index in range(1, 6)]
    _model_a_mse = [0.48, 0.42, 0.51, 0.45, 0.47]
    _model_b_mse = [0.44, 0.46, 0.43, 0.49, 0.41]
    _fold_frame = pd.DataFrame(
        [
            {
                "CV round": f"Round {round_index}",
                "Fold": f"Fold {fold_index}",
                "Role": "Validation fold" if fold_index == round_index else "Training folds",
                "Model A label": f"A MSE {model_a_mse:.2f}" if fold_index == round_index else "",
                "Model B label": f"B MSE {model_b_mse:.2f}" if fold_index == round_index else "",
            }
            for round_index, (model_a_mse, model_b_mse) in enumerate(zip(_model_a_mse, _model_b_mse), start=1)
            for fold_index in range(1, 6)
        ]
        + [
            {
                "CV round": f"Round {round_index}",
                "Fold": "Held-out test",
                "Role": "Held-out test set",
                "Model A label": "",
                "Model B label": "",
                "Test label": "Test",
            }
            for round_index in range(1, 6)
        ]
    )
    _fold_frame["Test label"] = _fold_frame.get("Test label", "").fillna("")
    _validation_frame = _fold_frame[_fold_frame["Role"] == "Validation fold"].copy()
    _test_frame = _fold_frame[_fold_frame["Role"] == "Held-out test set"].copy()
    _base = alt.Chart(_fold_frame).mark_rect(stroke="#FFFFFF", strokeWidth=1.4).encode(
        x=alt.X("Fold:N", sort=_fold_order, title="Data block"),
        y=alt.Y("CV round:N", sort=_round_order, title="CV round"),
        color=alt.Color(
            "Role:N",
            scale=alt.Scale(
                domain=["Training folds", "Validation fold", "Held-out test set"],
                range=["#D9E1E7", cv_color, holdout_color],
            ),
            legend=alt.Legend(title="Role"),
        ),
        tooltip=["CV round:N", "Fold:N", "Role:N"],
    )
    _model_a_labels = alt.Chart(_validation_frame).mark_text(
        dy=-8,
        fontSize=12,
        fontWeight="bold",
        color="#143D2A",
    ).encode(
        x=alt.X("Fold:N", sort=_fold_order),
        y=alt.Y("CV round:N", sort=_round_order),
        text="Model A label:N",
    )
    _model_b_labels = alt.Chart(_validation_frame).mark_text(
        dy=8,
        fontSize=12,
        fontWeight="bold",
        color="#143D2A",
    ).encode(
        x=alt.X("Fold:N", sort=_fold_order),
        y=alt.Y("CV round:N", sort=_round_order),
        text="Model B label:N",
    )
    _test_labels = alt.Chart(_test_frame).mark_text(
        fontSize=12,
        fontWeight="bold",
        color="#4B3B08",
    ).encode(
        x=alt.X("Fold:N", sort=_fold_order),
        y=alt.Y("CV round:N", sort=_round_order),
        text="Test label:N",
    )
    _cv_visual = finish_chart(
        (_base + _model_a_labels + _model_b_labels + _test_labels).properties(
            title="How 5-fold cross-validation averages validation MSE"
        ),
        width=650,
        height=175,
    )
    _mean_a = sum(_model_a_mse) / len(_model_a_mse)
    _mean_b = sum(_model_b_mse) / len(_model_b_mse)
    _layout = mo.vstack(
        [
            section_md(
                "9. Cross-Validation: Averaging Validation Folds",
                "Each green cell is the validation fold for that round; the gold column is the held-out test set.",
                (
                    "The numbers inside the green cells are hypothetical validation MSEs for two candidate models, A and B. We do **not** choose the model from one green cell. We average each candidate's validation MSE across the folds, then choose the model with the lower average validation MSE.\n\n"
                    "The gold held-out test column is not part of the CV average. It waits until after model selection."
                ),
            ),
            _cv_visual,
            mo.md(
                f"Example averages: Model A = `{_mean_a:.2f}`, Model B = `{_mean_b:.2f}`. The average, not any single fold, is the CV estimate used for model comparison."
            ),
        ],
        gap=0.24,
    )
    _layout
    return


@app.cell
def s8_controls(counter_button, mo):
    s8_n = mo.ui.slider(10, 220, value=10, step=10, label="Development sample size N")
    s8_recreate = counter_button(label="Recreate data", kind="success")
    return s8_n, s8_recreate


@app.cell
def s8_section(
    alt,
    build_selection_data,
    cv_color,
    evaluate_cv_curves,
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
    s8_n,
    s8_recreate,
    section_md,
    selection_degrees,
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
        degree_grid=selection_degrees,
        split_seed=_seed + 9,
        train_frac=0.50,
    )
    _split_b, _summary_b = evaluate_validation_curves(
        x_dev=_x_dev,
        y_dev=_y_dev,
        degree_grid=selection_degrees,
        split_seed=_seed + 41,
        train_frac=0.50,
    )
    _cv_frame = evaluate_cv_curves(_x_dev, _y_dev, selection_degrees, n_splits=5, seed=_seed + 73)
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
    _plot_frame = _comparison.copy()
    _plot_frame["Display MSE"] = np.maximum(_plot_frame["MSE"], 1e-3)
    _rules = pd.DataFrame(
        {
            "Degree": [int(_summary_a["chosen_degree"]), int(_summary_b["chosen_degree"]), _cv_degree],
            "Curve": ["Validation split A", "Validation split B", "5-fold CV"],
        }
    )
    _chart = alt.Chart(_plot_frame).mark_line(point=True, strokeWidth=2.6).encode(
        x=alt.X("Degree:Q", scale=alt.Scale(domain=[0, 10]), axis=alt.Axis(tickMinStep=1), title="Polynomial degree"),
        y=alt.Y("Display MSE:Q", title="Selection error (log scale)", scale=alt.Scale(type="log")),
        color=alt.Color(
            "Curve:N",
            scale=alt.Scale(
                domain=["Validation split A", "Validation split B", "5-fold CV"],
                range=[split_a_color, split_b_color, cv_color],
            ),
            legend=alt.Legend(title="Selection method"),
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
            legend=alt.Legend(title="Selection method"),
        ),
    )
    _chart = finish_chart(_chart + _rules_chart, height=250)
    _sidebar = sidebar(
        widgets=[s8_n, s8_recreate],
        metrics=[
            ("Split A chose", f"Degree {int(_summary_a['chosen_degree'])}"),
            ("Split B chose", f"Degree {int(_summary_b['chosen_degree'])}"),
            ("5-fold CV chose", f"Degree {_cv_degree}"),
            ("Held-out test MSE after CV", f"{_final_test_mse:.2f}"),
        ],
    )
    _layout = mo.vstack(
        [
            section_md(
                "Cross-Validation for Stability",
                "Now compare one-split validation choices with the 5-fold CV choice.",
                (
                    "Cross-validation is the remedy for instability in model comparison. Instead of letting one validation split carry the whole decision, we average validation MSE across folds and then choose the degree, from 0 through 10, with the lowest average error.\n\n"
                    "Cross-validation is not a replacement for the held-out test set. After choosing the degree, we retrain on all development data and use the held-out test set once for a final check."
                ),
            ),
            two_col(_sidebar, _chart),
            note_md("Cross-validation does not make noise disappear, but it usually reduces the chance that one arbitrary split dominates the complexity decision."),
            takeaway_md("Use validation or cross-validation to choose complexity, retrain on all development data, and save the held-out test set for one final check."),
            questions_md(
                [
                    "Why do we compare models using average validation MSE across folds?",
                    "After CV chooses the degree, what role is left for the held-out test MSE?",
                ]
            ),
        ],
        gap=0.30,
    )
    _layout
    return


if __name__ == "__main__":
    app.run()
