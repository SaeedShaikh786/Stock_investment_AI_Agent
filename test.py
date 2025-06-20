import os
import requests
import yfinance as yf
import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
import plotly.graph_objects as go
import ssl
import certifi
import logging # This was in your original script
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import base64
# import yfinance as yf # Already imported
# import requests # Already imported
from requests.packages.urllib3.exceptions import InsecureRequestWarning

from dotenv import load_dotenv
load_dotenv()  # This will load the .env file


# Disable warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Patch yfinance to not verify SSL
session = requests.Session()
session.verify = False
# ticker = yf.Ticker("AAPL") # This was in your original, but might not be necessary globally
# ticker._session = session # This patches a specific instance


os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context
# ssl._create_default_https_context = ssl._create_unverified_context  # Suggested code change (already in your original)

# Set environment variable for Google API
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

# Check if API key is available
if not os.getenv("GOOGLE_API_KEY"):
    st.warning("Please set the GOOGLE_API_KEY environment variable in your .env file.")

# Function to fetch and compare stock data
def compare_stocks(symbols):
    data = {}
    for symbol in symbols:
        if not symbol.strip(): # Avoid processing empty strings
            continue
        try:
            # Fetch stock data
            stock = yf.Ticker(symbol)
            hist = stock.history(period="6mo")  # Fetch last 6 months' data

            if hist.empty:
                print(f"No data found for {symbol}, skipping it.")
                continue  # Skip this ticker if no data found

            # Calculate overall % change
            data[symbol] = hist['Close'].pct_change().sum()

        except Exception as e:
            print(f"Could not retrieve data for {symbol}. Reason: {str(e)}")
            continue  # Skip this ticker if an error occurs
    return data

# Define the Market Analyst Agent
market_analyst = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Analyzes and compares stock performance over time.",
    instructions=[
        "Retrieve and compare stock performance from Yahoo Finance.",
        "Calculate percentage change over a 6-month period.",
        "Rank stocks based on their relative performance."
    ],
    show_tool_calls=True,
    markdown=True
)

# Function to get market analysis
def get_market_analysis(symbols):
    performance_data = compare_stocks(symbols)

    if not performance_data:
        return "No valid stock data found for the given symbols."

    analysis = market_analyst.run(f"Compare these stock performances: {performance_data}")
    return analysis.content

# ------------------------------ ---------------------------- #

def get_company_info(symbol):
    stock = yf.Ticker(symbol)
    return {
        "name": stock.info.get("longName", "N/A"),
        "sector": stock.info.get("sector", "N/A"),
        "market_cap": stock.info.get("marketCap", "N/A"),
        "summary": stock.info.get("longBusinessSummary", "N/A"),
    }

def get_company_news(symbol):
    stock = yf.Ticker(symbol)
    news = stock.news[:5]  # Get latest 5 news articles
    return news

company_researcher = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Fetches company profiles, financials, and latest news.",
    instructions=[
        "Retrieve company information from Yahoo Finance.",
        "Summarize latest company news relevant to investors.",
        "Provide sector, market cap, and business overview."
    ],
    markdown=True
)

def get_company_analysis(symbol):
    info = get_company_info(symbol)
    news = get_company_news(symbol)
    response = company_researcher.run(
        f"Provide an analysis for {info['name']} in the {info['sector']} sector.\n"
        f"Market Cap: {info['market_cap']}\n"
        f"Summary: {info['summary']}\n"
        f"Latest News: {news}"
    )
    return response.content

# ----------------------------- Stock strategist agent --------------------------- #
stock_strategist = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Provides investment insights and recommends top stocks.",
    instructions=[
        "Analyze stock performance trends and company fundamentals.",
        "Evaluate risk-reward potential and industry trends.",
        "Provide top stock recommendations for investors."
    ],
    markdown=True
)

# This function uses 'stocks_symbols' which is defined later from sidebar input.
# Ensure it's passed correctly if this function is called independently.
# For now, assuming it refers to the global 'stocks_symbols' from the Streamlit app flow.
def get_stock_recommendations(current_symbols): # Changed parameter name for clarity
    market_analysis = get_market_analysis(current_symbols)
    data = {}
    for symbol in current_symbols: # Use the passed parameter
        if symbol.strip(): # Ensure symbol is not empty
            data[symbol] = get_company_analysis(symbol)
    recommendations = stock_strategist.run(
        f"Based on the market analysis: {market_analysis}, and company news {data}"
        f"which stocks would you recommend for investment?"
    )
    return recommendations.content

# -------------------------------- Team Lead agent --------------------------------- #
team_lead = Agent(
    model=Gemini(id="gemini-2.0-flash-exp"),
    description="Aggregates stock analysis, company research, and investment strategy.",
    instructions=[
        "Compile stock performance, company analysis, and recommendations.",
        "Ensure all insights are structured in an investor-friendly report.",
        "Rank the top stocks based on combined analysis."
    ],
    markdown=True
)

def get_final_investment_report(current_symbols): # Changed parameter name
    market_analysis = get_market_analysis(current_symbols)
    # Ensure only valid symbols are passed to company analysis
    company_analyses = [get_company_analysis(symbol) for symbol in current_symbols if symbol.strip()]
    stock_recommendations = get_stock_recommendations(current_symbols)

    final_report = team_lead.run(
        f"Market Analysis:\n{market_analysis}\n\n"
        f"Company Analyses:\n{company_analyses}\n\n"
        f"Stock Recommendations:\n{stock_recommendations}\n\n"
        f"Provide the full analysis of each stock with Fundamentals and market news."
        f"Generate a final ranked list in ascending order on which should I buy."
    )
    return final_report.content


# Streamlit page configuration
st.set_page_config(page_title="AI Investment Strategist", layout="wide")

# Main content title and header
st.markdown("""
    <h1 style="text-align: center; color: #003D62;">AI Investment Strategist</h1>
    <h3 style="text-align: center; color: #6c757d;">Leveraging AI for Market Insight Reports and Comparisons.
</h3>
""", unsafe_allow_html=True)

# -- 🖼 Logo with Rounded Corners in Sidebar
def get_base64_logo(img_path):
    try:
        img = Image.open(img_path)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except FileNotFoundError:
        print(f"Warning: Logo image not found at {img_path}")
        return None
 
logo_base64 = get_base64_logo("static/Hoonartek-V25-White-Color.png")

# --- Sidebar Styling ---
st.markdown(f"""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');

        html, body, [class*="css"], h1, h2, h3, h4, h5, h6, p {{
            font-family: 'Poppins', sans-serif;
        }}

        [data-testid="stSidebar"] {{
            background: linear-gradient(135deg, #003D62, #3497BA);
            color: white;
            padding: 20px;
            width: 320px !important;
        }}

        [data-testid="stSidebar"] input {{
            border-radius: 8px;
            padding: 0.5em;
        }}

        /* MODIFIED: Original button style with new transition */
        [data-testid="stSidebar"] button {{
            background-color: transparent !important;
            color: #E0E0E0 !important; /* Slightly lighter grey for better visibility on gradient */
            font-weight: bold;
            border: 2px solid #E0E0E0 !important;
            border-radius: 8px; 
            padding: 0.6em 1em;
            margin-top: 10px;
            transition: background-color 0.3s ease, color 0.3s ease, border-color 0.3s ease !important; /* MODIFIED: Added smooth transition */
        }}

        /* MODIFIED: New hover effect for the button */
        [data-testid="stSidebar"] button:hover {{
            background-color: #A7D8DE !important;  /* Light teal/blue background */
            color: #003D62 !important;             /* Dark blue text for contrast */
            border: 2px solid #A7D8DE !important;  /* Matching border color */
        }}

        [data-testid="stSidebar"] h2 {{
            color: white;
            font-size: 1.4rem;
            margin-bottom: 0.5em;
        }}

        /* MODIFIED: Ensured Poppins font and color for list items in expander too */
        [data-testid="stSidebar"] p, [data-testid="stSidebar"] li {{
            color: #f0f8ff;
            font-size: 0.95rem;
            line-height: 1.5;
        }}
        /* MODIFIED: Styling for expander */
        [data-testid="stExpander"] summary {{
            font-size: 0.95rem;
            color: #f0f8ff; /* Light color for expander header */
        }}
        [data-testid="stExpander"] {{
            border: 1px solid #f0f8ff !important; /* Light border for expander */
            border-radius: 8px;
            margin-top: 10px;
            margin-bottom: 10px; /* Added some margin below expander */
        }}
    </style>
""", unsafe_allow_html=True)

if logo_base64: # Only display if logo was loaded
    st.sidebar.markdown(
        f"""
        <div style="text-align: center;">
            <img src="data:image/png;base64,{logo_base64}" style="width: 220px; border-radius: 0px; margin-bottom: 20px;" />
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.sidebar.warning("Logo image not found. Please check the path: static/Hoonartek-V25-White-Color.png")


# -- 🧠 Use Case Description
st.sidebar.title("Use Case Details")
st.sidebar.markdown(
    """
AI Stock Report Generator provides real-time analysis of U.S. and Indian stock markets. Users enter stock symbols (e.g., AAPL, MRF.NS) to receive AI-curated reports featuring key metrics, price trends, and performance comparisons. It offers a quick, intuitive way to evaluate and compare multiple stocks."""
)

st.sidebar.subheader("Model Name:\nGemini-2.0-flash-exp")

# --- Stock Symbol Input ---
input_symbols = st.sidebar.text_input("Enter Stock Symbols")
 

# Parse the stock symbols input
# MODIFIED: Filter out empty strings that might result from "AAPL," or ",,"
stocks_symbols = [symbol.strip().upper() for symbol in input_symbols.split(",") if symbol.strip()]

# MODIFIED: Replaced the old markdown help with an expander
with st.sidebar.expander("Example Stock Symbols"):
    st.markdown("""
    Enter symbols separated by commas. Tickers are generally uppercase.
    
    **US Stocks (NASDAQ/NYSE):**
    *   `AAPL` (Apple Inc.)
    *   `MSFT` (Microsoft Corp.)
    *   `GOOGL` (Alphabet Inc. Class A)
    *   `NVDA` (NVIDIA Corporation)

    **Indian Stocks (NSE - append .NS):**
    *   `RELIANCE.NS` (Reliance Industries Ltd.)
    *   `TCS.NS` (Tata Consultancy Services Ltd.)
    *   `INFY.NS` (Infosys Ltd.)
    *   `HDFCBANK.NS` (HDFC Bank Ltd.)
    """)

# Generate Investment Report button
if st.sidebar.button("Generate Report"):
    # MODIFIED: Simplified check for empty stocks_symbols list
    if not stocks_symbols:
        st.sidebar.warning("Please enter at least one stock symbol.")
    else:
        with st.spinner(f"Generating report for: {', '.join(stocks_symbols)}..."):
            # Generate the final report
            report = get_final_investment_report(stocks_symbols) # Pass the parsed symbols

            # Display the report
            st.subheader("Investment Report")
            st.markdown(report)

            # Check if the report content indicates issues before trying to plot
            if report and "No valid stock data" not in report and "Could not retrieve data" not in report:
                st.info("This report provides detailed insights, including market performance, company analysis, and investment recommendations.")

                # Interactive Stock Performance Chart
                st.markdown("### 📊 Stock Performance (6-Months)")
                try:
                    # yf.download can handle a list of symbols
                    # Using progress=False to avoid potential issues in some environments
                    stock_data_close = yf.download(stocks_symbols, period="6mo", progress=False)['Close']
                    
                    if stock_data_close.empty:
                        st.warning("Could not download any stock data for the chart. Please check the symbols.")
                    else:
                        fig = go.Figure()
                        # If only one stock, stock_data_close might be a Series, not DataFrame
                        if isinstance(stock_data_close, yf.pd.Series): # yfinance uses pandas, so pd is available via yf.pd
                             fig.add_trace(go.Scatter(x=stock_data_close.index, y=stock_data_close, mode='lines', name=stocks_symbols[0]))
                        else: # Multiple stocks, it's a DataFrame
                            for symbol_to_plot in stocks_symbols: # Iterate through requested symbols
                                if symbol_to_plot in stock_data_close.columns: # Check if data exists for this symbol
                                     fig.add_trace(go.Scatter(x=stock_data_close.index, y=stock_data_close[symbol_to_plot], mode='lines', name=symbol_to_plot))
                                else:
                                    st.warning(f"No chart data available for {symbol_to_plot}. It might have been skipped due to an error or no historical data.")

                        fig.update_layout(title="Stock Performance Over the Last 6 Months",
                                          xaxis_title="Date",
                                          yaxis_title="Price", # Generalizing as it can be mixed currency
                                          template="plotly_dark")
                        st.plotly_chart(fig)
                except Exception as e:
                    st.error(f"An error occurred while generating the stock performance chart: {e}")
            elif report: # Report was generated but might contain warnings from AI/data fetching
                st.warning("Report generated, but there might be issues with some data. Please review.")
            else: # Should not happen if AI agents return content, but as a fallback
                st.error("Failed to generate the report. Please check the input symbols and try again.")
