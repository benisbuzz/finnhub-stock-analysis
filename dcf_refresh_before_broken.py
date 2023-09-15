import finnhub
import fredpy as fp
from typing import Any, Optional, Literal
import requests
import datetime as dt
import time
import numpy as np
import pandas as pd
from dataclasses import dataclass
import os
from dotenv import load_dotenv

load_dotenv()

fp.api_key = os.getenv("FREDPY_API_KEY")
finnhub_client = finnhub.Client(api_key=os.getenv("FINNHUB_API_KEY"))
all_bonds = requests.get(
    f"https://finnhub.io/api/v1/bond/list?token={finnhub_client.api_key}"
).json()

FinnhubDataType = list[dict[str, Any]]


@dataclass
class CompanyBalanceSheet:
    accountsReceivables: pd.Series
    accountsPayable: pd.Series
    inventory: pd.Series
    totalDebt: pd.Series
    netDebt: pd.Series


@dataclass
class CompanyIncomeStatement:
    ebit: pd.Series
    provisionforIncomeTaxes: pd.Series
    pretaxIncome: pd.Series
    revenue: pd.Series
    costOfGoodsSold: pd.Series


@dataclass
class CompanyCashflowStatement:
    depreciationAmortization: pd.Series
    capex: pd.Series
    cashInterestPaid: pd.Series


@dataclass
class CompanyFinancials:
    balance_sheet: CompanyBalanceSheet
    income_statement: CompanyIncomeStatement
    cashflow_statement: CompanyCashflowStatement


@dataclass
class CompanyEstimates:
    revenue: pd.Series
    ebitda: pd.Series
    ebit: pd.Series


@dataclass
class AllData:
    financials: CompanyFinancials
    estimates: CompanyEstimates


def _get_data_series(datasets: FinnhubDataType, key: str) -> pd.Series:
    if key == "inventory" and all(key not in dataset for dataset in datasets):
        return None
    elif all(key not in dataset for dataset in datasets):
        raise KeyError(f"Key does not exist (enough): {key}")

    datasets = sorted(datasets, key=lambda d: d["year"])
    return pd.Series(
        data=[dataset.get(key, np.nan) for dataset in datasets],
        index=[dataset["year"] for dataset in datasets],
        dtype="float64",
    ).interpolate(limit_direction="backward")


def get_financials(ticker: str) -> CompanyEstimates:
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
            if _get_data_series(dataset, "inventory") is not None:
                balance_sheets = CompanyBalanceSheet(
                    **{
                        key: _get_data_series(dataset, key)
                        for key in [
                            "accountsReceivables",
                            "accountsPayable",
                            "totalDebt",
                            "netDebt",
                            "inventory",
                        ]
                    }
                )
            else:
                years = _get_data_series(dataset, "accountsReceivables").index
                balance_sheet_items = {
                    key: _get_data_series(dataset, key)
                    for key in [
                        "accountsReceivables",
                        "accountsPayable",
                        "totalDebt",
                        "netDebt",
                    ]
                }
                balance_sheet_items.update(
                    {"inventory": pd.Series(np.zeros(len(years)), years)}
                )
                balance_sheets = CompanyBalanceSheet(**balance_sheet_items)
        elif statement == "income_statements":
            income_statements = CompanyIncomeStatement(
                **{
                    key: _get_data_series(dataset, key)
                    for key in [
                        "ebit",
                        "provisionforIncomeTaxes",
                        "pretaxIncome",
                        "revenue",
                        "costOfGoodsSold",
                    ]
                }
            )
        else:
            cashflow_statements = CompanyCashflowStatement(
                **{
                    key: _get_data_series(dataset, key)
                    for key in ["depreciationAmortization", "capex", "cashInterestPaid"]
                }
            )

    return CompanyFinancials(balance_sheets, income_statements, cashflow_statements)


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
) -> CompanyEstimates:
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
    return CompanyEstimates(**equal_estimates)


def _get_future_data(historical_data: pd.Series, estimates: pd.Series) -> pd.Series:
    """Return estimates for future years only"""
    for year in estimates.index:
        if year <= historical_data.index[-1]:
            estimates = estimates.drop(year)
    return estimates


def _get_common_years(series_to_filter: dict[str, pd.Series]) -> dict[str, pd.Series]:
    common_years = sorted(
        list(
            set.intersection(
                *map(set, [series.index for series in series_to_filter.values()])
            )
        )
    )
    return {
        key: series.filter(items=common_years)
        for key, series in series_to_filter.items()
    }


def get_ebitda(
    financial_statements: CompanyFinancials, estimates: CompanyEstimates
) -> pd.Series:
    """Return time series of historical and future ebitda"""
    historical_ebit = financial_statements.income_statement.ebit
    historical_depreciation_amortization = (
        financial_statements.cashflow_statement.depreciationAmortization
    )
    historical_ebitda = historical_ebit + historical_depreciation_amortization
    future_ebitda = _get_future_data(historical_ebitda, estimates.ebitda)
    return pd.concat([historical_ebitda, future_ebitda])


def get_tax_rate(financial_statements: CompanyFinancials) -> np.float64:
    """Return average effective tax rate in the last 3 years"""
    filtered_data = _get_common_years(
        {
            "tax_payed": financial_statements.income_statement.provisionforIncomeTaxes,
            "pretax_income": financial_statements.income_statement.pretaxIncome,
        }
    )
    if len(filtered_data["tax_payed"]) < 3:
        return np.clip(
            np.mean(filtered_data["tax_payed"] / filtered_data["pretax_income"]),
            0.1,
            0.25,
        )
    else:
        return np.clip(
            np.mean(
                filtered_data["tax_payed"][-3:] / filtered_data["pretax_income"][-3:]
            ),
            0.1,
            0.25,
        )


def get_tax_paid(
    financial_statements: CompanyFinancials, estimates: CompanyEstimates
) -> pd.Series:
    historical_data = financial_statements.income_statement.provisionforIncomeTaxes * -1
    future_ebit = _get_future_data(historical_data, estimates.ebit)
    tax_rate = get_tax_rate(financial_statements)
    return pd.concat([historical_data, future_ebit * tax_rate * -1])


def get_mean_cost_of_goods_sold(financial_statements: CompanyFinancials) -> np.float64:
    filtered_data = _get_common_years(
        {
            "revenue": financial_statements.income_statement.revenue,
            "cost_of_goods": financial_statements.income_statement.costOfGoodsSold,
        }
    )
    if len(filtered_data["revenue"] < 5):
        return np.mean(filtered_data["cost_of_goods"] / filtered_data["revenue"])
    else:
        return np.mean(
            filtered_data["cost_of_goods"][-5:] / filtered_data["revenue"][-5:]
        )


def get_median_years(financial_statements: CompanyFinancials) -> dict[str, np.float64]:
    filtered_data = _get_common_years(
        {
            "revenue": financial_statements.income_statement.revenue,
            "cost_of_goods": financial_statements.income_statement.costOfGoodsSold,
            "inventory": financial_statements.balance_sheet.inventory,
            "accounts_receivable": financial_statements.balance_sheet.accountsReceivables,
            "accounts_payable": financial_statements.balance_sheet.accountsPayable,
        }
    )
    result = {}
    for key, value in filtered_data.items():
        if key not in ["revenue", "cost_of_goods", "accounts_receivable"]:
            result.update({key: np.median(value / filtered_data["cost_of_goods"])})
        elif key == "accounts_receivable":
            result.update({key: np.median(value / filtered_data["revenue"])})
        else:
            pass

    return result


def get_forecasted_balance_sheet(
    financial_statements: CompanyFinancials, estimates: CompanyEstimates
) -> dict[str, pd.Series]:
    median_years = get_median_years(financial_statements)
    revenue_estimates = estimates.revenue
    cost_of_goods_estimates = (
        get_mean_cost_of_goods_sold(financial_statements) * revenue_estimates
    )
    result = {}
    for key, value in median_years.items():
        if key == "inventory":
            historical_inventory = financial_statements.balance_sheet.inventory
            result.update(
                {
                    key: pd.concat(
                        [
                            historical_inventory,
                            _get_future_data(
                                historical_inventory, cost_of_goods_estimates
                            )
                            * value,
                        ]
                    )
                }
            )
        elif key == "accounts_receivable":
            historical_accounts_receivable = (
                financial_statements.balance_sheet.accountsReceivables
            )
            result.update(
                {
                    key: pd.concat(
                        [
                            historical_accounts_receivable,
                            _get_future_data(
                                historical_accounts_receivable, revenue_estimates
                            )
                            * value,
                        ]
                    )
                }
            )
        else:
            historical_accounts_payable = (
                financial_statements.balance_sheet.accountsPayable
            )
            result.update(
                {
                    key: pd.concat(
                        [
                            historical_accounts_payable,
                            _get_future_data(
                                historical_accounts_payable, cost_of_goods_estimates
                            )
                            * value,
                        ]
                    )
                }
            )

    return result


def get_changes_owc(
    financial_statements: CompanyFinancials, estimates: CompanyEstimates
) -> pd.Series:
    forecasted_balance_sheet = get_forecasted_balance_sheet(
        financial_statements, estimates
    )
    return (
        (
            forecasted_balance_sheet["accounts_receivable"]
            + forecasted_balance_sheet["inventory"]
            - forecasted_balance_sheet["accounts_payable"]
        )
        .diff()
        .dropna()
    )


def get_capex(
    financial_statements: CompanyFinancials, estimates: CompanyEstimates
) -> pd.Series:
    historical_capex = financial_statements.cashflow_statement.capex
    capex_to_sales = np.mean(
        historical_capex / financial_statements.income_statement.revenue
    )
    return pd.concat(
        [
            historical_capex,
            _get_future_data(historical_capex, estimates.revenue) * capex_to_sales,
        ]
    )


def get_unlevered_fcf(
    financial_statements: CompanyFinancials, estimates: CompanyEstimates
) -> pd.Series:
    filtered_data = _get_common_years(
        {
            "ebitda": get_ebitda(financial_statements, estimates),
            "tax_paid": get_tax_paid(financial_statements, estimates),
            "changes_owc": get_changes_owc(financial_statements, estimates),
            "capex": get_capex(financial_statements, estimates),
        }
    )
    return (
        filtered_data["ebitda"]
        + filtered_data["tax_paid"]
        + filtered_data["changes_owc"]
        + filtered_data["capex"]
    )


def get_capm(ticker: str) -> float:
    risk_free = fp.series("FEDFUNDS").data[-1] / 100
    beta = finnhub_client.company_basic_financials(ticker, "all")["metric"]["beta"]
    return risk_free + beta * max(0, 0.06 - risk_free)


def get_bond_cost_of_debt(ticker: str) -> np.float64:
    """Return average yield to maturity for a company given stock ticker"""
    short_isin = finnhub_client.company_profile(symbol=ticker)["isin"][:8]
    company_bonds = [
        profile for profile in all_bonds if short_isin in profile["isin"][:8]
    ]
    start = int(time.mktime((dt.datetime.today() - dt.timedelta(days=100)).timetuple()))
    ytm = []
    if company_bonds == []:
        return None
    else:
        for profile in company_bonds:
            if (
                dt.datetime.strptime(profile["maturityDate"], "%Y-%m-%d")
                - dt.datetime.today()
            ).days > 0 and profile["coupon"] is not None:
                coupon = profile["coupon"]
                years = (
                    dt.datetime.strptime(profile["maturityDate"], "%Y-%m-%d")
                    - dt.datetime.today()
                ).days / 365
                end = int(time.mktime(dt.datetime.today().timetuple()))
                price_data = finnhub_client.bond_price(profile["isin"], start, end)
                price = price_data["c"][-1] if price_data["s"] == "ok" else np.nan
                if price is not np.nan:
                    ytm.append(
                        ((coupon + ((100 - price) / years)) / ((100 + price) / 2))
                    )
                else:
                    pass
    return np.mean(ytm)


def get_bs_cost_of_debt(financial_statements: CompanyFinancials) -> np.float64:
    filtered_data = _get_common_years(
        {
            "interest_paid": financial_statements.cashflow_statement.cashInterestPaid,
            "total_debt": financial_statements.balance_sheet.totalDebt,
        }
    )
    return np.mean(
        (filtered_data["interest_paid"] / filtered_data["total_debt"])
        .replace(np.inf, np.nan)
        .dropna()
    )


def get_wacc(ticker: str, financial_statements: CompanyFinancials) -> float:
    cost_of_equity = get_capm(ticker)
    bond_cost_of_debt = get_bond_cost_of_debt(ticker)
    cost_of_debt = (
        bond_cost_of_debt
        if bond_cost_of_debt is not None
        else get_bs_cost_of_debt(financial_statements)
    )
    mkt_cap = finnhub_client.company_profile2(symbol=ticker)["marketCapitalization"]
    total_debt = financial_statements.balance_sheet.totalDebt.iloc[-1]
    return (mkt_cap / (total_debt + mkt_cap)) * cost_of_equity + (
        total_debt / (total_debt + mkt_cap) * cost_of_debt
    )


def get_terminal_growth(
    financial_statements: CompanyFinancials, estimates: CompanyEstimates
) -> float:
    ufcf = get_unlevered_fcf(financial_statements, estimates)
    cagr = (ufcf.iloc[-1] / ufcf.iloc[0]) ** (1 / (len(ufcf) - 1)) - 1
    if cagr < 0:
        terminal_growth = 0.01
    elif cagr > 0.05:
        terminal_growth = 0.05
    else:
        terminal_growth = cagr
    return terminal_growth


def get_terminal_value(
    ticker: str, financial_statements: CompanyFinancials, estimates: CompanyEstimates
) -> float:
    final_ufcf = get_unlevered_fcf(financial_statements, estimates).iloc[-1]
    terminal_growth = get_terminal_growth(financial_statements, estimates)
    wacc = get_wacc(ticker, financial_statements)
    return final_ufcf * ((1 + terminal_growth) / (wacc - terminal_growth))


def get_implied_ev_to_ebitda(
    ticker: str, financial_statements: CompanyFinancials, estimates: CompanyEstimates
) -> float:
    return (
        get_terminal_value(ticker, financial_statements, estimates)
        / get_ebitda(financial_statements, estimates).iloc[-1]
    )


def get_present_value(
    ticker: str, financial_statements: CompanyFinancials, estimates: CompanyEstimates
) -> float:
    current_reporting_year = dt.date.today().year - 1
    ufcf = pd.Series(
        [
            value
            for key, value in get_unlevered_fcf(financial_statements, estimates).items()
            if key > current_reporting_year
        ],
        [
            key
            for key, _ in get_unlevered_fcf(financial_statements, estimates).items()
            if key > current_reporting_year
        ],
    )
    wacc = get_wacc(ticker, financial_statements)
    discount_factors = pd.Series(
        [
            (1 / (1 + wacc)) ** (future_year - current_reporting_year)
            for future_year in sorted(ufcf.index)
        ],
        [future_year for future_year in sorted(ufcf.index)],
    )
    pv_ufcf = ufcf * discount_factors
    pv_terminal_value = (
        get_terminal_value(ticker, financial_statements, estimates)
        * discount_factors.iloc[-1]
    )
    return sum(pv_ufcf) + pv_terminal_value


def get_valuation(
    ticker: str, financial_statements: CompanyFinancials, estimates: CompanyEstimates
) -> float:
    pv_of_cashflows = get_present_value(ticker, financial_statements, estimates)
    equity_value = pv_of_cashflows - financial_statements.balance_sheet.netDebt.iloc[-1]
    shares_outstanding = finnhub_client.company_profile(symbol=ticker)[
        "shareOutstanding"
    ]
    return equity_value / shares_outstanding
