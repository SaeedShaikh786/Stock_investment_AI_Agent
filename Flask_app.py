from flask import Flask, render_template, request, jsonify
import os
import yfinance as yf
from agno.agent import Agent
from agno.models.google import Gemini
import plotly.graph_objects as go
import json
import ssl
import certifi
import requests
from plotly.utils import PlotlyJSONEncoder
import pandas as pd # <-- Import pandas

app = Flask(__name__)

# Set environment variable for Google API
# IMPORTANT SECURITY WARNING: Never hardcode API keys directly in source code.
# Use environment variables or a secret management system in production.
# Example: os.environ.get("GOOGLE_API_KEY", "YOUR_DEFAULT_KEY_IF_NEEDED")
# For this example, using the provided key, but strongly advise against it.
os.environ["GOOGLE_API_KEY"] = "AIzaSyCr35hxFrpVsbNWgqOwU6PwmkpwLmO2dJA" # <-- WARNING: Hardcoded API Key

# SSL Certificate Fix
# Check if running in an environment where certifi path is needed
try:
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    # Note: Using ssl._create_unverified_context disables important security checks.
    # This should only be used if you understand the risks and cannot resolve the underlying SSL issue.
    # A better approach is to ensure your system's certificate store is up-to-date.
    # ssl._create_default_https_context = ssl._create_unverified_context # <-- Commented out for safety unless absolutely necessary
except ImportError:
    print("Certifi not found. Skipping SSL context modifications.")


# Disable SSL verification for yfinance and agno (Use with caution!)
# This is generally not recommended for production environments due to security risks.
# Consider fixing the root cause of SSL verification failures if possible.
yf_session = requests.Session()
yf_session.verify = False # Disable SSL verification for yfinance
yf.utils.requests = yf_session

agno_session = requests.Session()
agno_session.verify = False # Disable SSL verification for agno

# Suppress only the single InsecureRequestWarning from urllib3 needed for verify=False
requests.packages.urllib3.disable_warnings(requests.packages.urllib3.exceptions.InsecureRequestWarning)


# Define AI agents (assuming Gemini class and Agent class are correctly defined elsewhere)
# Make sure the agno library uses the agno_session if needed for HTTPS requests.
# Depending on agno's implementation, you might need to pass the session or configure it globally.

market_analyst = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"), # Ensure this model ID is correct and available
    description="Analyzes and compares stock performance over time.",
    instructions=[
        "Retrieve and compare stock performance from Yahoo Finance.",
        "Calculate percentage change over a 6-month period.",
        "Rank stocks based on their relative performance."
    ],
    markdown=True)

company_researcher = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Fetches company profiles, financials, and latest news.",
    instructions=[
        "Retrieve company information from Yahoo Finance.",
        "Summarize latest company news relevant to investors.",
        "Provide sector, market cap, and business overview."
    ],
    markdown=True)

stock_strategist = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Provides investment insights and recommends top stocks.",
    instructions=[
        "Analyze stock performance trends and company fundamentals.",
        "Evaluate risk-reward potential and industry trends.",
        "Provide top stock recommendations for investors."
    ],
    markdown=True)

team_lead = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Aggregates stock analysis, company research, and investment strategy.",
    instructions=[
        "Compile stock performance, company analysis, and recommendations.",
        "Ensure all insights are structured in an investor-friendly report.",
        "Rank the top stocks based on combined analysis."
    ],
    markdown=True)

# Helper functions
def compare_stocks(symbols):
    """Calculates the cumulative percentage change over 6 months for given stock symbols."""
    data = {}
    # Standardize symbols to uppercase
    symbols_upper = [s.strip().upper() for s in symbols if s.strip()]
    if not symbols_upper:
        return {}

    try:
        # Download historical data
        hist = yf.download(symbols_upper, period="6mo", progress=False)
        if hist.empty:
            print(f"Warning: No historical data found for symbols: {symbols_upper}")
            return {}

        # Select 'Close' prices
        close_prices = hist['Close']

        # Handle cases: single symbol (Series) vs multiple symbols (DataFrame)
        if isinstance(close_prices, pd.Series):
            # If only one symbol was successful, convert Series to DataFrame
            close_prices = pd.DataFrame({symbols_upper[0]: close_prices})

        # Calculate cumulative percentage change for each column (symbol)
        for symbol in close_prices.columns:
             # Use pct_change().sum() - Be aware this isn't true cumulative return, but sum of daily changes.
             # A better measure might be (last_price / first_price) - 1
            # perf = (close_prices[symbol].iloc[-1] / close_prices[symbol].iloc[0]) - 1 if not close_prices[symbol].empty and close_prices[symbol].iloc[0] != 0 else 0
            perf = close_prices[symbol].pct_change().sum() # Keep original logic if intended
            if pd.notna(perf): # Check if performance is not NaN
                 data[symbol] = perf

    except Exception as e:
        print(f"Error comparing stocks {symbols_upper}: {e}")
        # Optionally return partial data or handle specific errors
    return data

def get_market_analysis(symbols):
    performance_data = compare_stocks(symbols)
    if not performance_data:
        return "Could not retrieve or calculate performance data for the given symbols."
    # Create a formatted string for the agent
    perf_string = ", ".join([f"{sym}: {perf:.2%}" for sym, perf in performance_data.items()])
    prompt = f"Analyze and compare the 6-month stock performance (sum of daily % change) for these symbols: {perf_string}. Rank them based on this performance."
    try:
        analysis = market_analyst.run(prompt)
        return analysis.content
    except Exception as e:
        print(f"Error running market analyst agent: {e}")
        return f"Error generating market analysis: {e}"


def get_company_info(symbol):
    """Fetches basic company information."""
    try:
        stock = yf.Ticker(symbol)
        # Use .get() with default values to avoid errors if keys are missing
        info = stock.info
        return {
            "name": info.get("longName", symbol), # Fallback to symbol if name missing
            "sector": info.get("sector", "N/A"),
            "industry": info.get("industry", "N/A"),
            "market_cap": info.get("marketCap", "N/A"),
            "summary": info.get("longBusinessSummary", "No summary available."),
        }
    except Exception as e:
        print(f"Error fetching info for {symbol}: {e}")
        return {
            "name": symbol,
            "sector": "Error", "industry": "Error", "market_cap": "Error", "summary": f"Could not fetch data: {e}",
        }

def get_company_news(symbol):
    """Fetches recent company news headlines."""
    try:
        stock = yf.Ticker(symbol)
        news = stock.news
        # Return titles and links, ensuring structure is simple for the agent
        return [{"title": item.get('title', 'N/A'), "link": item.get('link', '#')} for item in news[:5]] # Get top 5
    except Exception as e:
        print(f"Error fetching news for {symbol}: {e}")
        return [{"title": f"Could not fetch news: {e}", "link": "#"}]

def get_company_analysis(symbol):
    """Generates analysis for a single company using its info and news."""
    info = get_company_info(symbol)
    news = get_company_news(symbol) # News is now a list of dicts

    # Format news for the prompt
    news_summary = "\n".join([f"- {item['title']}" for item in news])
    if not news_summary:
        news_summary = "No recent news found."

    prompt = (
        f"Provide an investment analysis for {info['name']} ({symbol}).\n"
        f"Sector: {info['sector']}, Industry: {info['industry']}\n"
        f"Market Cap: {info['market_cap']}\n"
        f"Business Summary: {info['summary']}\n\n"
        f"Recent News Headlines:\n{news_summary}\n\n"
        f"Based on this information, summarize the company's current standing for a potential investor."
    )
    try:
        response = company_researcher.run(prompt)
        return response.content
    except Exception as e:
        print(f"Error running company researcher agent for {symbol}: {e}")
        return f"Error generating company analysis for {symbol}: {e}"

def get_stock_recommendations(symbols):
    """Generates stock recommendations based on market and company analyses."""
    # Ensure symbols are valid before proceeding
    valid_symbols = [s.strip().upper() for s in symbols if s.strip()]
    if not valid_symbols:
        return "No valid symbols provided for recommendations."

    market_analysis = get_market_analysis(valid_symbols)
    # Generate analysis for each company *concurrently* if possible (or sequentially)
    # For simplicity, doing it sequentially here:
    company_analyses_dict = {}
    for symbol in valid_symbols:
         # Get a concise summary for the recommendation prompt
         # Note: get_company_analysis already calls the agent.
         # Avoid calling the agent twice if possible. Maybe get_company_analysis should return structured data?
         # For now, we'll call it again, but it's inefficient.
         company_summary = get_company_analysis(symbol) # Re-calls agent
         company_analyses_dict[symbol] = company_summary # Storing the full analysis text

    # Format company summaries for the strategist prompt
    company_summaries_text = "\n\n".join([f"--- Analysis for {sym} ---\n{analysis}" for sym, analysis in company_analyses_dict.items()])

    prompt = (
        f"You are a Stock Strategist. Based on the following market performance analysis and individual company analyses, please provide investment recommendations.\n\n"
        f"Overall Market Context (6-Month Performance Ranking):\n{market_analysis}\n\n"
        f"Individual Company Analyses:\n{company_summaries_text}\n\n"
        f"Instructions: Evaluate the risk-reward potential, consider industry trends, and recommend which of these stocks ({', '.join(valid_symbols)}) seem most promising for investment, explaining your reasoning. Rank your recommendations if possible."
    )
    try:
        recommendations = stock_strategist.run(prompt)
        return recommendations.content
    except Exception as e:
        print(f"Error running stock strategist agent: {e}")
        return f"Error generating stock recommendations: {e}"


def get_final_investment_report(symbols):
    """Compiles the final investment report using all analyses."""
    valid_symbols = [s.strip().upper() for s in symbols if s.strip()]
    if not valid_symbols:
        return "No valid symbols provided for the final report."

    print(f"Generating final report for: {valid_symbols}") # Debug print

    # Reuse previously defined functions
    market_analysis = get_market_analysis(valid_symbols)
    print("Market analysis fetched.") # Debug print

    company_analyses = []
    for symbol in valid_symbols:
        print(f"Fetching company analysis for {symbol}...") # Debug print
        analysis = get_company_analysis(symbol)
        company_analyses.append({"symbol": symbol, "analysis": analysis})
    print("Company analyses fetched.") # Debug print

    stock_recommendations = get_stock_recommendations(valid_symbols) # This might re-run analyses internally, check for efficiency
    print("Stock recommendations fetched.") # Debug print


    # Format company analyses for the final report prompt
    company_analyses_text = "\n\n".join([f"### Analysis for {item['symbol']}\n{item['analysis']}" for item in company_analyses])

    prompt = (
        f"You are the Team Lead. Compile a comprehensive investment report for the following stocks: {', '.join(valid_symbols)}.\n\n"
        f"## Market Performance Overview (Last 6 Months)\n{market_analysis}\n\n"
        f"## Detailed Company Analyses\n{company_analyses_text}\n\n"
        f"## Investment Strategy & Recommendations\n{stock_recommendations}\n\n"
        f"Instructions: Structure these insights into a clear, investor-friendly report. Ensure the final output includes the market comparison, detailed analysis for each company (fundamentals, news impact), and the final ranked recommendations with justifications. Combine all information cohesively."
    )
    try:
        print("Running team lead agent...") # Debug print
        final_report = team_lead.run(prompt)
        print("Team lead agent finished.") # Debug print
        return final_report.content
    except Exception as e:
        print(f"Error running team lead agent: {e}")
        return f"Error generating final investment report: {e}"

# Flask routes
@app.route('/')
def index():
    return render_template('index.html') # Make sure index.html exists in a 'templates' folder

@app.route('/generate_report', methods=['POST'])
def generate_report():
    symbols_input = request.form.get('symbols', '').split(',')
    # Clean and standardize symbols (e.g., uppercase, remove whitespace)
    symbols = [symbol.strip().upper() for symbol in symbols_input if symbol.strip()]

    if not symbols:
        return jsonify({"error": "Please provide at least one stock symbol."}), 400

    try:
        # --- Download Data for Chart ---
        # Use group_by='ticker' for cleaner column structure, especially with errors
        stock_data_full = yf.download(symbols, period="6mo", progress=False, group_by='ticker')

        # Check if the download returned anything
        if stock_data_full.empty:
             # Try downloading one by one to see which ones work, provide partial data?
             # For now, return error if initial bulk download fails completely.
            return jsonify({"error": f"Could not download any data for symbols: {', '.join(symbols)}"}), 500

        # --- Filter for Successful Downloads and 'Close' Price ---
        stock_data_close = pd.DataFrame()
        valid_symbols_for_chart = []

        for symbol in symbols:
            if symbol in stock_data_full and 'Close' in stock_data_full[symbol]:
                # Check if the 'Close' column has non-NaN data
                if not stock_data_full[symbol]['Close'].isnull().all():
                    stock_data_close[symbol] = stock_data_full[symbol]['Close']
                    valid_symbols_for_chart.append(symbol)

        # Check if we have any valid data to plot
        if stock_data_close.empty:
            return jsonify({"error": "No valid closing price data found for the provided symbols to generate a chart."}), 500

        print(f"Data for chart downloaded for: {valid_symbols_for_chart}") # Debug print

        # --- Generate Plotly Chart ---
        fig = go.Figure()
        # Iterate over the columns of the DataFrame containing CLOSE prices
        for symbol_to_plot in stock_data_close.columns: # Iterate columns of the filtered DF
            fig.add_trace(go.Scatter(
                x=stock_data_close.index,
                y=stock_data_close[symbol_to_plot], # Access the column directly
                mode='lines',
                name=symbol_to_plot # Use the column name as the trace name
            ))

        fig.update_layout(
            title="Stock Performance Over the Last 6 Months",
            xaxis_title="Date",
            yaxis_title="Price (in USD)",
            template="plotly_dark", # Example template
            legend_title_text='Symbols'
        )

        # Serialize chart to JSON
        chart_json = json.dumps(fig, cls=PlotlyJSONEncoder)
        print("Chart JSON generated.") # Debug print

        # --- Generate Report using symbols that had chart data ---
        # It's slightly inefficient if report generation needs different data,
        # but ensures report matches plotted symbols.
        report = get_final_investment_report(valid_symbols_for_chart)
        print("Report content generated.") # Debug print

        return jsonify({"report": report, "chart": chart_json})

    except Exception as e:
        import traceback
        print("Error in /generate_report endpoint:")
        print(traceback.format_exc()) # Print full traceback to console
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    # Add host='0.0.0.0' to make it accessible on your network if needed
    # Use port other than default 5000 if it's occupied
    app.run(debug=True) # debug=True is helpful for development, disable for production
    
