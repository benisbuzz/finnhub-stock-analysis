from matplotlib import pyplot as plt
from dataclasses import dataclass
import finnhub
import pandas as pd
import numpy as np
from typing import Optional, Literal
import datetime as dt
import os
from dotenv import load_dotenv

load_dotenv()

finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))
FinnhubDataType = list[dict[str, any]]


@dataclass
class IncomeStatement:
    ebitda: pd.Series
    revenue: pd.Series


@dataclass
class CashflowStatement:
    fcf: pd.Series
    capex: pd.Series


@dataclass
class BalanceSheet:
    totalEquity: pd.Series
    netDebt: pd.Series


@dataclass
class Financials:
    balance_sheet: BalanceSheet
    income_statement: IncomeStatement
    cashflow_statement: CashflowStatement


@dataclass
class Estimates:
    revenue: pd.Series
    ebitda: pd.Series
    ebit: pd.Series


def _get_data_series(datasets: FinnhubDataType, key: str) -> pd.Series:
    if all(key not in dataset for dataset in datasets):
        raise KeyError(f"Key does not exist (enough): {key}")

    datasets = sorted(datasets, key=lambda d: d["year"])
    return pd.Series(
        data=[dataset.get(key, np.nan) for dataset in datasets],
        index=[dataset["year"] for dataset in datasets],
        dtype="float64",
    )


def _get_ebitda(ic: FinnhubDataType, cf: FinnhubDataType) -> pd.Series:
    return _get_data_series(ic, "ebit") + _get_data_series(
        cf, "depreciationAmortization"
    )


def get_financials(ticker: str) -> Financials:
    financials_raw = {
        "balance_sheets": finnhub_client.financials(ticker, "bs", "annual")[
            "financials"
        ],
        "income_statements": finnhub_client.financials(ticker, "ic", "annual")[
            "financials"
        ],
        "cashflow_statements": finnhub_client.financials(ticker, "cf", "annual")[
            "financials"
        ],
    }

    if any(value is None for value in financials_raw.values()):
        raise RuntimeError(f"Financial statements missing for {ticker}")

    for statement, dataset in financials_raw.items():
        if statement == "balance_sheets":
            balance_sheets = BalanceSheet(
                **{
                    key: _get_data_series(dataset, key)
                    for key in ["netDebt", "totalEquity"]
                }
            )
        elif statement == "income_statements":
            income_statements = IncomeStatement(
                **{
                    "revenue": _get_data_series(dataset, "revenue"),
                    "ebitda": _get_ebitda(
                        dataset, financials_raw["cashflow_statements"]
                    ),
                }
            )
        else:
            cashflow_statements = CashflowStatement(
                **{key: _get_data_series(dataset, key) for key in ["fcf", "capex"]}
            )
    return Financials(balance_sheets, income_statements, cashflow_statements)


def _get_future_length(lst: list[tuple[int, float]]) -> int:
    current_year = dt.date.today().year
    return len({year for year, _ in lst if year > current_year})


def _get_median(numerator: pd.Series, denominator: pd.Series) -> np.float64:
    years = set(numerator.index) & set(denominator.index)
    nfiltered = numerator.filter(items=years)
    dfiltered = denominator.filter(items=years)
    return np.median(nfiltered / dfiltered)


def get_estimates(
    ticker: str, estimate_level: Optional[Literal["Low", "High", "Avg"]] = "Avg"
) -> Estimates:
    revenue_raw = finnhub_client.company_revenue_estimates(ticker, "annual")["data"]
    ebitda_raw = finnhub_client.company_ebitda_estimates(ticker, "annual")["data"]
    ebit_raw = finnhub_client.company_ebit_estimates(ticker, "annual")["data"]

    estimates = {
        "revenue": _get_data_series(revenue_raw, f"revenue{estimate_level}"),
        "ebitda": _get_data_series(ebitda_raw, f"ebitda{estimate_level}"),
        "ebit": _get_data_series(ebit_raw, f"ebit{estimate_level}"),
    }

    if any(value is None for value in estimates.values()):
        raise RuntimeError(f"at least one estimate missing for {ticker}")

    longest_future = max(
        estimates, key=lambda et: _get_future_length(estimates[et].items())
    )
    longest_future_estimate = estimates[longest_future]
    other_estimates = [estimates[key] for key in estimates if key is not longest_future]

    for other_estimate in other_estimates:
        median = _get_median(other_estimate, longest_future_estimate)
        for year in sorted(longest_future_estimate.index):
            if year not in other_estimate.index:
                other_estimate.at[year] = longest_future_estimate.at[year] * median

    equal_estimates = {
        key: value.sort_index() / 1_000_000 for key, value in estimates.items()
    }
    return Estimates(**equal_estimates)


def _get_future_data(historical_data: pd.Series, estimates: pd.Series) -> pd.Series:
    """Return estimates for future years only"""
    for year in estimates.index:
        if year <= historical_data.index[-1]:
            estimates = estimates.drop(year)
    return pd.concat(
        [historical_data.filter(items=[historical_data.index[-1]]), estimates]
    )


def get_summary(ticker: str) -> None:
    financials = get_financials(ticker)
    estimates = get_estimates(ticker)
    fig, ax = plt.subplots(2, 3, figsize=(20, 10))

    fig.suptitle(f"{ticker.upper()} Summary")
    fig.tight_layout(pad=5)

    ax[0, 0].plot(financials.balance_sheet.totalEquity, "-o")
    ax[0, 0].set_title("Equity")

    ax[0, 1].plot(financials.balance_sheet.netDebt, "-o")
    ax[0, 1].set_title("Net Debt")

    ax[0, 2].plot(financials.income_statement.revenue, "-o")
    ax[0, 2].plot(
        _get_future_data(financials.income_statement.revenue, estimates.revenue), "-o"
    )
    ax[0, 2].set_title("Revenue")

    ax[1, 0].plot(financials.income_statement.ebitda, "-o")
    ax[1, 0].plot(
        _get_future_data(financials.income_statement.ebitda, estimates.ebitda), "-o"
    )
    ax[1, 0].set_title("EBITDA")

    ax[1, 1].plot(financials.cashflow_statement.capex, "-o")
    ax[1, 1].set_title("CAPEX")

    ax[1, 2].plot(financials.cashflow_statement.fcf, "-o")
    ax[1, 2].set_title("FCF")
