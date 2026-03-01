limport time
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import yfinance as yf
from dash import Dash, Input, Output, State, ctx, dcc, html, no_update
from edgar import Company, set_identity

# Important: Set your email here to comply with SEC EDGAR access policies.
# This is required to use the edgar package for fetching Form 4 filings.
set_identity("email@email.com")

# Cache settings
CACHE_TTL_SECONDS = 900
DEFAULT_FILING_LIMIT = 200
FORM4_CACHE = {}
PRICE_CACHE = {}

# Visual theme colors for chart + layout
COLORS = {
    "bg": "#0b2545",
    "card_bg": "#12355b",
    "card_border": "#1f4d7a",
    "text": "#ffffff",
    "muted": "#ffffff",
    "sales": "#FAB700",
    "acq": "#01F9DC",
    "price": "#8ecae6",
}

CARD_STYLE = {
    "backgroundColor": COLORS["card_bg"],
    "border": f"1px solid {COLORS['card_border']}",
    "borderRadius": "0.75rem",
    "color": COLORS["text"],
}


def _is_light_color(hex_color):
    value = (hex_color or "").lstrip("#")
    if len(value) != 6:
        return False
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return luminance > 0.62


def build_container_style(bg_color, text_color=None):
    text_color = text_color or ("#111111" if _is_light_color(bg_color) else "#ffffff")
    return {
        "background": bg_color,
        "minHeight": "100vh",
        "color": text_color,
    }


def build_card_style(card_bg, text_color=None):
    is_light = _is_light_color(card_bg)
    border = (
        "1px solid rgba(0, 0, 0, 0.2)"
        if is_light
        else "1px solid rgba(255, 255, 255, 0.2)"
    )
    text_color = text_color or ("#111111" if is_light else "#ffffff")
    return {
        "backgroundColor": card_bg,
        "border": border,
        "borderRadius": "0.75rem",
        "color": text_color,
    }


def hex_to_rgba(hex_color, alpha):
    value = (hex_color or "").lstrip("#")
    if len(value) != 6:
        return f"rgba(250, 183, 0, {alpha})"
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


# Custom exception for data source errors to provide clearer error handling and messaging
class DataSourceError(Exception):
    pass


# Helper functions for caching, CIK resolution, data fetching,
# figure building, and error classification are defined below to keep the main callback logic
# clean and focused on orchestrating the dashboard updates.
def _get_cached(cache_store, key):
    cached = cache_store.get(key)
    if not cached:
        return None
    age = time.time() - cached["ts"]
    if age > CACHE_TTL_SECONDS:
        cache_store.pop(key, None)
        return None
    return cached["value"]


# Helper function to set cache with current timestamp
def _set_cached(cache_store, key, value):
    cache_store[key] = {"ts": time.time(), "value": value}


# Helper function to resolve CIK from ticker using
# SEC's company_tickers.json endpoint
def _resolve_cik_from_ticker(ticker):
    ticker = (ticker or "").strip().upper()
    if not ticker:
        raise DataSourceError("Ticker is empty.")

    try:
        cik_lookup = requests.get(
            "https://www.sec.gov/files/company_tickers.json",
            headers={"User-Agent": "email@email.com"},
            timeout=15,
        ).json()
    except Exception as exc:
        raise DataSourceError(f"Failed to download SEC CIK lookup: {exc}") from exc

    for company in cik_lookup.values():
        if str(company.get("ticker", "")).upper() == ticker:
            return str(company["cik_str"]).zfill(10)

    raise DataSourceError(f"Ticker not found in SEC company_tickers.json: {ticker}.")


# Main function to fetch Form 4 data for a given ticker and filing
# limit, with caching support
def fetch_form4_dataframe(ticker, filing_limit=DEFAULT_FILING_LIMIT):
    cache_key = (ticker, filing_limit)
    cached = _get_cached(FORM4_CACHE, cache_key)
    if cached is not None:
        return cached.copy(deep=True), True

    try:
        cik = _resolve_cik_from_ticker(ticker)
        company = Company(cik)
        filings = company.get_filings(form="4").head(filing_limit)
    except Exception as exc:
        raise DataSourceError(f"SEC lookup failed for {ticker}: {exc}") from exc

    frames = []
    for filing in filings:
        try:
            frames.append(filing.obj().to_dataframe().fillna(""))
        except Exception:
            continue

    if not frames:
        raise DataSourceError(f"No Form 4 filings were returned for {ticker}.")

    df = pd.concat(frames, ignore_index=True)
    _set_cached(FORM4_CACHE, cache_key, df)
    return df.copy(deep=True), False


# Function to build a monthly summary of sales (S) and acquisitions (A) from the Form 4 data,
def build_monthly(df, metric_mode="count"):
    if df.empty or "Code" not in df.columns or "Date" not in df.columns:
        return pd.DataFrame(columns=["S", "A"])

    working = df[["Code", "Date"]].copy()
    use_shares = metric_mode == "shares" and "Shares" in df.columns
    if use_shares:
        working["Shares"] = pd.to_numeric(
            df["Shares"].astype(str).str.replace(",", "", regex=False), errors="coerce"
        ).fillna(0.0)
    working["Date"] = pd.to_datetime(working["Date"], errors="coerce")
    working = working.dropna(subset=["Date"])
    working = working[working["Code"].isin(["S", "A"])]

    if working.empty:
        return pd.DataFrame(columns=["S", "A"])

    working["Month"] = working["Date"].dt.to_period("M").astype(str)
    if use_shares:
        monthly = (
            working.groupby(["Month", "Code"])["Shares"].sum().unstack().sort_index()
        )
        monthly = monthly.fillna(0.0).astype(float)
    else:
        monthly = working.groupby(["Month", "Code"]).size().unstack().sort_index()
        monthly = monthly.fillna(0).astype(int)

    if "S" not in monthly.columns:
        monthly["S"] = 0.0 if use_shares else 0
    if "A" not in monthly.columns:
        monthly["A"] = 0.0 if use_shares else 0

    return monthly[["S", "A"]]


# Function to fetch monthly closing prices from Yahoo Finance for the
# date range covered by the Form 4 data, with caching support
def fetch_monthly_prices(ticker, monthly):
    if monthly.empty:
        return pd.DataFrame(columns=["Month", "YahooClose"]), False

    month_idx = pd.to_datetime(monthly.index, format="%Y-%m")
    start_date = month_idx.min().strftime("%Y-%m-%d")
    end_date = (month_idx.max() + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")

    cache_key = (ticker, start_date, end_date)
    cached = _get_cached(PRICE_CACHE, cache_key)
    if cached is not None:
        return cached.copy(deep=True), True

    try:
        price_df = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=True,
        )
    except Exception as exc:
        raise DataSourceError(f"Yahoo price lookup failed for {ticker}: {exc}") from exc

    if price_df.empty:
        raise DataSourceError(
            f"Yahoo returned no price data for {ticker} in range {start_date} to {end_date}."
        )

    close_series = price_df["Close"]
    if isinstance(close_series, pd.DataFrame):
        close_series = close_series.iloc[:, 0]

    monthly_close = close_series.resample("ME").last().dropna()
    monthly_price_df = (
        monthly_close.to_frame(name="YahooClose")
        .assign(Month=lambda d: d.index.to_period("M").astype(str))
        .reset_index(drop=True)
    )

    _set_cached(PRICE_CACHE, cache_key, monthly_price_df)
    return monthly_price_df.copy(deep=True), False


# Function to build the Plotly figure showing monthly sales and acquisitions,
# along with optional trend lines and price overlay, based on the provided data
# and metric mode (count vs shares)
def build_figure(
    monthly,
    monthly_price_df,
    ticker,
    metric_mode="count",
    s_color=None,
    a_color=None,
    text_color=None,
    card_bg=None,
):
    metric_label = "Shares" if metric_mode == "shares" else "Count"
    s_color = s_color or COLORS["sales"]
    a_color = a_color or COLORS["acq"]
    text_color = text_color or COLORS["text"]
    card_bg = card_bg or COLORS["card_bg"]
    s_bar_color = hex_to_rgba(s_color, 0.45)
    a_bar_color = hex_to_rgba(a_color, 0.45)
    if monthly.empty:
        fig = go.Figure()
        fig.update_layout(
            title=f"No Form 4 S/A {metric_label.lower()} data found for {ticker}",
            xaxis_title="Month",
            yaxis_title=metric_label,
            template="plotly_white",
        )
        return fig

    monthly_df = monthly.reset_index().melt(
        id_vars="Month", value_vars=["S", "A"], var_name="Code", value_name="Count"
    )

    fig = px.bar(
        monthly_df,
        x="Month",
        y="Count",
        color="Code",
        barmode="group",
        title=f"Insider Monthly Sales (S) vs Acquisitions (A) {metric_label} - {ticker}",
        color_discrete_map={"S": s_bar_color, "A": a_bar_color},
    )

    monthly_s = monthly["S"].reset_index(name="SCount")
    if len(monthly_s) >= 2:
        x_idx = np.arange(len(monthly_s))
        s_coeffs = np.polyfit(x_idx, monthly_s["SCount"], 1)
        fig.add_trace(
            go.Scatter(
                x=monthly_s["Month"],
                y=np.polyval(s_coeffs, x_idx),
                mode="lines",
                name=f"Trend Sales ({metric_label})",
                line=dict(color=s_color, width=3, dash="dash"),
            )
        )

    monthly_a = monthly["A"].reset_index(name="ACount")
    if len(monthly_a) >= 2:
        x_idx = np.arange(len(monthly_a))
        a_coeffs = np.polyfit(x_idx, monthly_a["ACount"], 1)
        fig.add_trace(
            go.Scatter(
                x=monthly_a["Month"],
                y=np.polyval(a_coeffs, x_idx),
                mode="lines",
                name=f"Trend Acquisitions ({metric_label})",
                line=dict(color=a_color, width=3, dash="dash"),
            )
        )

    if not monthly_price_df.empty:
        fig.add_trace(
            go.Scatter(
                x=monthly_price_df["Month"],
                y=monthly_price_df["YahooClose"],
                mode="lines",
                name=f"{ticker} Month Closing Price",
                line=dict(color=COLORS["price"], width=3),
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis2=dict(
                title=f"{ticker} Monthly Closing Price",
                overlaying="y",
                side="right",
                showgrid=False,
            )
        )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=card_bg,
        plot_bgcolor=card_bg,
        font=dict(color=text_color),
        yaxis_title=metric_label,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# Function to classify error messages into user-friendly categories for display in the
# dashboard status message area
def classify_error_message(exc):
    text = str(exc)
    lowered = text.lower()
    if "429" in text or "rate" in lowered or "too many" in lowered:
        return "Rate limited by SEC or Yahoo. Wait a minute and retry."
    if "no form 4 filings" in lowered:
        return text
    if "no price data" in lowered or "possibly delisted" in lowered:
        return "No Yahoo price data found for this ticker/date range."
    if "not found" in lowered or "invalid" in lowered:
        return "Ticker not recognized. Check symbol and retry."
    return text


# Main function to load Form 4 data, build the dashboard figure, and calculate summary metrics
# for a given ticker and filing limit, with error handling and caching support
def load_ticker_dashboard(
    ticker,
    filing_limit,
    metric_mode="count",
    s_color=None,
    a_color=None,
    text_color=None,
    card_bg=None,
):
    df, filings_from_cache = fetch_form4_dataframe(ticker, filing_limit)
    monthly = build_monthly(df, metric_mode=metric_mode)
    if monthly.empty:
        raise DataSourceError(
            f"No S/A transactions found in Form 4 filings for {ticker}."
        )

    monthly_price_df, prices_from_cache = fetch_monthly_prices(ticker, monthly)

    fig = build_figure(
        monthly,
        monthly_price_df,
        ticker,
        metric_mode=metric_mode,
        s_color=s_color,
        a_color=a_color,
        text_color=text_color,
        card_bg=card_bg,
    )
    if metric_mode == "shares":
        total_s = f"{int(round(monthly['S'].sum())):,}" if not monthly.empty else "0"
        total_a = f"{int(round(monthly['A'].sum())):,}" if not monthly.empty else "0"
    else:
        total_s = f"{int(monthly['S'].sum()):,}" if not monthly.empty else "0"
        total_a = f"{int(monthly['A'].sum()):,}" if not monthly.empty else "0"
    latest_close = (
        f"${monthly_price_df['YahooClose'].iloc[-1]:,.2f}"
        if not monthly_price_df.empty
        else "N/A"
    )

    return fig, total_s, total_a, latest_close, filings_from_cache, prices_from_cache


# Helper function to create an empty Plotly figure with a given title, used for initial
# state and error cases
def empty_figure(title, text_color=None, card_bg=None):
    text_color = text_color or COLORS["text"]
    card_bg = card_bg or COLORS["card_bg"]
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        paper_bgcolor=card_bg,
        plot_bgcolor=card_bg,
        font=dict(color=text_color),
    )
    return fig


# Initialize the Dash app with Bootstrap styling and define the layout, which includes input
# fields for ticker and filing limit, metric mode selection, summary cards for total sales/acquisitions
# and latest close price, and a graph area for the monthly S/A chart.
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])

# Define the app layout with input fields, summary cards, and graph area
app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H5(
                                "Recent Counts of Insider Stock Sales and Acquisitions from SEC Form 4 Filings",
                                className="card-title mb-0",
                            ),
                            html.P(
                                "Enter a ticker to pull live Form 4 (EDGAR) and price (Yahoo) data",
                                id="subtitle-text",
                                className="mb-0",
                                style={"color": COLORS["muted"]},
                            ),
                        ]
                    ),
                    id="header-card",
                    className="shadow-sm",
                    style=CARD_STYLE,
                ),
                width=12,
            ),
            className="mb-3",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Input(
                        id="ticker-input",
                        type="text",
                        value="",
                        placeholder="Enter ticker (e.g., AAPL)",
                    ),
                    md=4,
                ),
                dbc.Col(
                    dcc.Dropdown(
                        id="limit-input",
                        options=[
                            {"label": "50 filings", "value": 50},
                            {"label": "100 filings", "value": 100},
                            {"label": "200 filings", "value": 200},
                        ],
                        value=None,
                        placeholder="Select number of recent filings",
                        clearable=False,
                        style={"color": "black"},
                    ),
                    md=3,
                ),
                dbc.Col(
                    dbc.RadioItems(
                        id="metric-mode",
                        options=[
                            {"label": "Count", "value": "count"},
                            {"label": "Shares", "value": "shares"},
                        ],
                        value="count",
                        inline=True,
                    ),
                    md=3,
                ),
                dbc.Col(
                    dbc.Button(
                        "Pull Data",
                        id="pull-button",
                        color="primary",
                        n_clicks=0,
                        disabled=True,
                    ),
                    md="auto",
                ),
                dbc.Col(
                    [
                        html.Div(
                            id="updating-msg",
                            className="fw-semibold",
                            style={"color": COLORS["text"]},
                        ),
                        html.Div(id="status-msg", style={"color": COLORS["text"]}),
                    ],
                    md=4,
                ),
            ],
            className="mb-3 align-items-center",
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [html.H6("Total Sales"), html.H4("-", id="total-s")]
                        ),
                        id="sales-card",
                        className="shadow-sm",
                        style=CARD_STYLE,
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [html.H6("Total Acquisitions"), html.H4("-", id="total-a")]
                        ),
                        id="acq-card",
                        className="shadow-sm",
                        style=CARD_STYLE,
                    ),
                    md=4,
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H6("Latest Close", id="latest-close-label"),
                                html.H4("-", id="latest-close"),
                            ]
                        ),
                        id="close-card",
                        className="shadow-sm",
                        style=CARD_STYLE,
                    ),
                    md=4,
                ),
            ],
            className="mb-3",
        ),
        dbc.Row(
            dbc.Col(
                dbc.Card(
                    dbc.CardBody(
                        dcc.Graph(
                            id="filings-chart",
                            figure=empty_figure("Enter a ticker and click Pull Data"),
                        )
                    ),
                    id="chart-card",
                    className="shadow-sm",
                    style=CARD_STYLE,
                ),
                width=12,
            )
        ),
        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Label("S Color", className="mb-1"),
                        dbc.Input(
                            id="s-color",
                            type="color",
                            value=COLORS["sales"],
                            style={"width": "3rem", "height": "1.8rem", "padding": "0"},
                        ),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        html.Label("A Color", className="mb-1"),
                        dbc.Input(
                            id="a-color",
                            type="color",
                            value=COLORS["acq"],
                            style={"width": "3rem", "height": "1.8rem", "padding": "0"},
                        ),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        html.Label("Text Color", className="mb-1"),
                        dbc.Input(
                            id="text-color",
                            type="color",
                            value=COLORS["text"],
                            style={"width": "3rem", "height": "1.8rem", "padding": "0"},
                        ),
                    ],
                    md=2,
                ),
                dbc.Col(
                    [
                        html.Label("Dashboard Background", className="mb-1"),
                        dbc.Input(
                            id="dashboard-bg-color",
                            type="color",
                            value=COLORS["bg"],
                            style={"width": "3rem", "height": "1.8rem", "padding": "0"},
                        ),
                    ],
                    md=3,
                ),
                dbc.Col(
                    [
                        html.Label("Card Background", className="mb-1"),
                        dbc.Input(
                            id="card-bg-color",
                            type="color",
                            value=COLORS["card_bg"],
                            style={"width": "3rem", "height": "1.8rem", "padding": "0"},
                        ),
                    ],
                    md=3,
                ),
            ],
            className="mt-3 mb-3 align-items-end",
        ),
    ],
    id="app-container",
    fluid=True,
    className="py-3",
    style=build_container_style(COLORS["bg"]),
)


# Define the callback function that orchestrates the data fetching, processing, and figure building
@app.callback(
    Output("filings-chart", "figure"),
    Output("total-s", "children"),
    Output("total-a", "children"),
    Output("latest-close", "children"),
    Output("latest-close-label", "children"),
    Output("status-msg", "children"),
    Input("pull-button", "n_clicks"),
    Input("metric-mode", "value"),
    Input("s-color", "value"),
    Input("a-color", "value"),
    Input("text-color", "value"),
    Input("card-bg-color", "value"),
    State("ticker-input", "value"),
    State("limit-input", "value"),
    State("filings-chart", "figure"),
    running=[
        (
            Output("updating-msg", "children"),
            html.Span(
                [
                    dbc.Spinner(size="sm", color="primary", type="border"),
                    html.Span(
                        " Updating graph... pulling data and building chart.",
                        className="ms-2",
                    ),
                ],
                className="d-inline-flex align-items-center",
            ),
            "",
        ),
    ],
)

# The main callback function that handles button clicks, triggers data loading and processing,
def pull_and_render(
    _n_clicks,
    metric_mode,
    s_color,
    a_color,
    text_color,
    card_bg,
    ticker_value,
    limit_value,
    current_figure,
):
    ticker = (ticker_value or "").strip().upper()
    filing_limit = int(limit_value or DEFAULT_FILING_LIMIT)
    metric_mode = "shares" if metric_mode == "shares" else "count"
    text_color = text_color or COLORS["text"]
    card_bg = card_bg or COLORS["card_bg"]

    # If only style inputs changed, update chart appearance without re-pulling data.
    if ctx.triggered_id in {"text-color", "card-bg-color"} and current_figure:
        fig = go.Figure(current_figure)
        fig.update_layout(
            paper_bgcolor=card_bg,
            plot_bgcolor=card_bg,
            font=dict(color=text_color),
        )
        return fig, no_update, no_update, no_update, no_update, no_update

    if not ticker:
        return (
            empty_figure("No ticker provided", text_color=text_color, card_bg=card_bg),
            "-",
            "-",
            "-",
            "Latest Close",
            "Enter a ticker symbol.",
        )

    try:
        start = time.perf_counter()
        fig, total_s, total_a, latest_close, f_cached, p_cached = load_ticker_dashboard(
            ticker,
            filing_limit,
            metric_mode=metric_mode,
            s_color=s_color,
            a_color=a_color,
            text_color=text_color,
            card_bg=card_bg,
        )
        elapsed = time.perf_counter() - start

        cache_parts = []
        if f_cached:
            cache_parts.append("EDGAR cache")
        if p_cached:
            cache_parts.append("Yahoo cache")
        cache_suffix = f" ({', '.join(cache_parts)})" if cache_parts else ""

        return (
            fig,
            total_s,
            total_a,
            latest_close,
            f"{ticker} Latest Close",
            f"Loaded {ticker} with {metric_mode} view using last {filing_limit} Form 4 filings in {elapsed:.2f}s{cache_suffix}",
        )
    except Exception as exc:
        return (
            empty_figure(
                f"Unable to load data for {ticker}",
                text_color=text_color,
                card_bg=card_bg,
            ),
            "-",
            "-",
            "-",
            "Latest Close",
            f"Error: {classify_error_message(exc)}",
        )


@app.callback(
    Output("app-container", "style"),
    Output("header-card", "style"),
    Output("sales-card", "style"),
    Output("acq-card", "style"),
    Output("close-card", "style"),
    Output("chart-card", "style"),
    Output("subtitle-text", "style"),
    Output("updating-msg", "style"),
    Output("status-msg", "style"),
    Output("pull-button", "style"),
    Output("filings-chart", "style"),
    Input("dashboard-bg-color", "value"),
    Input("card-bg-color", "value"),
    Input("text-color", "value"),
)
def update_background_colors(dashboard_bg, card_bg, text_color):
    dashboard_bg = dashboard_bg or COLORS["bg"]
    card_bg = card_bg or COLORS["card_bg"]
    text_color = text_color or COLORS["text"]
    card_style = build_card_style(card_bg, text_color=text_color)
    shared_text_style = {"color": text_color}
    return (
        build_container_style(dashboard_bg, text_color=text_color),
        card_style,
        card_style,
        card_style,
        card_style,
        card_style,
        shared_text_style,
        shared_text_style,
        shared_text_style,
        {
            "color": text_color,
            "backgroundColor": card_bg,
            "borderColor": card_bg,
        },
        {"backgroundColor": card_bg, "borderRadius": "0.5rem"},
    )


@app.callback(
    Output("pull-button", "disabled"),
    Input("limit-input", "value"),
)
def toggle_pull_button(limit_value):
    return limit_value is None


# Run the Dash app in debug mode when this script is executed directly
if __name__ == "__main__":
    app.run(debug=True)
