import yfinance as yf
from datetime import datetime


class YFinanceFetcher:

    def get_company_info(self, ticker: str) -> dict:
        """Basic company metadata"""
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "ticker": ticker,
            "company_name": info.get("longName", ticker),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "market_cap": info.get("marketCap"),
            "employees": info.get("fullTimeEmployees"),
            "description": info.get("longBusinessSummary"),
            "website": info.get("website"),
            "exchange": info.get("exchange"),
        }

    def get_financial_metrics(self, ticker: str) -> dict:
        """
        Structured financial data — used to cross-verify
        numbers extracted from transcripts.
        """
        stock = yf.Ticker(ticker)
        info = stock.info

        return {
            "ticker": ticker,
            # Valuation
            "pe_ratio": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "price_to_book": info.get("priceToBook"),
            "price_to_sales": info.get("priceToSalesTrailing12Months"),
            "ev_to_ebitda": info.get("enterpriseToEbitda"),
            # Revenue & Profit
            "revenue_ttm": info.get("totalRevenue"),
            "gross_profit": info.get("grossProfits"),
            "operating_income": info.get("operatingIncome"),
            "net_income": info.get("netIncomeToCommon"),
            "ebitda": info.get("ebitda"),
            # Margins
            "gross_margin": info.get("grossMargins"),
            "operating_margin": info.get("operatingMargins"),
            "profit_margin": info.get("profitMargins"),
            # Per Share
            "eps_trailing": info.get("trailingEps"),
            "eps_forward": info.get("forwardEps"),
            # Growth
            "revenue_growth": info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            # Balance Sheet
            "total_cash": info.get("totalCash"),
            "total_debt": info.get("totalDebt"),
            "free_cashflow": info.get("freeCashflow"),
            # Analyst
            "analyst_target_price": info.get("targetMeanPrice"),
            "analyst_recommendation": info.get("recommendationKey"),
            "number_of_analysts": info.get("numberOfAnalystOpinions"),
            "fetched_at": datetime.utcnow().isoformat(),
        }

    def get_quarterly_financials(self, ticker: str) -> dict:
        """
        Quarter by quarter income statement, balance sheet, cash flow.
        Used for trend analysis across quarters.
        """
        stock = yf.Ticker(ticker)

        try:
            income_stmt = stock.quarterly_income_stmt
            balance_sheet = stock.quarterly_balance_sheet
            cash_flow = stock.quarterly_cashflow
        except Exception as e:
            print(f"Could not fetch quarterly financials for {ticker}: {e}")
            return {}

        def df_to_dict(df):
            if df is None or df.empty:
                return {}
            return {
                str(col): {
                    row: (
                        float(df.loc[row, col])
                        if df.loc[row, col] is not None
                        else None
                    )
                    for row in df.index
                }
                for col in df.columns
            }

        return {
            "ticker": ticker,
            "income_statement": df_to_dict(income_stmt),
            "balance_sheet": df_to_dict(balance_sheet),
            "cash_flow": df_to_dict(cash_flow),
            "fetched_at": datetime.utcnow().isoformat(),
        }

    def get_earnings_history(self, ticker: str) -> list[dict]:
        """
        Historical earnings — actual vs estimated EPS.
        Used to compute earnings surprise.
        """
        stock = yf.Ticker(ticker)

        try:
            earnings_dates = stock.earnings_dates
            if earnings_dates is None or earnings_dates.empty:
                return []
        except Exception as e:
            print(f"Could not fetch earnings history for {ticker}: {e}")
            return []

        records = []
        for date, row in earnings_dates.iterrows():
            records.append({
                "date": str(date.date()),
                "eps_estimate": row.get("EPS Estimate"),
                "eps_actual": row.get("Reported EPS"),
                "eps_surprise_pct": row.get("Surprise(%)"),
                "ticker": ticker,
            })

        return records

    def get_analyst_recommendations(self, ticker: str) -> list[dict]:
        """Buy / sell / hold recommendations over time"""
        stock = yf.Ticker(ticker)

        try:
            recs = stock.recommendations
            if recs is None or recs.empty:
                return []
        except Exception as e:
            print(f"Could not fetch recommendations for {ticker}: {e}")
            return []

        records = []
        for date, row in recs.iterrows():
            records.append({
                "date": str(date.date()),
                "firm": row.get("Firm"),
                "to_grade": row.get("To Grade"),
                "from_grade": row.get("From Grade"),
                "action": row.get("Action"),
                "ticker": ticker,
            })

        return records

    def get_price_history(
        self,
        ticker: str,
        period: str = "1y",
    ) -> list[dict]:
        """Historical OHLCV price data"""
        stock = yf.Ticker(ticker)

        try:
            hist = stock.history(period=period)
            if hist.empty:
                return []
        except Exception as e:
            print(f"Could not fetch price history for {ticker}: {e}")
            return []

        records = []
        for date, row in hist.iterrows():
            records.append({
                "date": str(date.date()),
                "open": round(float(row["Open"]), 4),
                "high": round(float(row["High"]), 4),
                "low": round(float(row["Low"]), 4),
                "close": round(float(row["Close"]), 4),
                "volume": int(row["Volume"]),
                "ticker": ticker,
            })

        return records

    def fetch_all(self, ticker: str) -> dict:
        """Fetch everything for a ticker in one call"""
        print(f"  Fetching yfinance data for {ticker}...")
        return {
            "company_info": self.get_company_info(ticker),
            "financial_metrics": self.get_financial_metrics(ticker),
            "quarterly_financials": self.get_quarterly_financials(ticker),
            "earnings_history": self.get_earnings_history(ticker),
            "analyst_recommendations": self.get_analyst_recommendations(ticker),
            "price_history": self.get_price_history(ticker),
        }
