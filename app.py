import os
import requests
import yfinance as yf
import streamlit as st
from agno.agent import Agent
from agno.models.google import Gemini
import plotly.graph_objects as go
import ssl
import certifi
import logging
from dotenv import load_dotenv
from PIL import Image
from io import BytesIO
import base64
import yfinance as yf
import requests
from requests.packages.urllib3.exceptions import InsecureRequestWarning

from dotenv import load_dotenv
load_dotenv()  # This will load the .env file


# Disable warnings
requests.packages.urllib3.disable_warnings(InsecureRequestWarning)

# Patch yfinance to not verify SSL
session = requests.Session()
session.verify = False
ticker = yf.Ticker("AAPL")
ticker._session = session


os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
ssl._create_default_https_context = ssl._create_unverified_context
ssl._create_default_https_context = ssl._create_unverified_context  # Suggested code change
# Disable SSL verification for yfinance
# Set environment variable for Google API
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")  # Get API key from .env

# Check if API key is available
if not os.getenv("GOOGLE_API_KEY"):
    st.warning("Please set the GOOGLE_API_KEY environment variable in your .env file.")

# Function to fetch and compare stock data
def compare_stocks(symbols):
    data = {}
    for symbol in symbols:
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
    model=Gemini(id="gemini-2.0-flash"),
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
    model=Gemini(id="gemini-2.0-flash"),
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
    model=Gemini(id="gemini-2.0-flash"),
    description="Provides investment insights and recommends top stocks.",
    instructions=[
        "Analyze stock performance trends and company fundamentals.",
        "Evaluate risk-reward potential and industry trends.",
        "Provide top stock recommendations for investors."
    ],
    markdown=True
)

def get_stock_recommendations(symbols):
    market_analysis = get_market_analysis(symbols)
    data = {}
    for symbol in stocks_symbols:
        data[symbol] = get_company_analysis(symbol)
    recommendations = stock_strategist.run(
        f"Based on the market analysis: {market_analysis}, and company news {data}"
        f"which stocks would you recommend for investment?"
    )
    return recommendations.content

# -------------------------------- Team Lead agent --------------------------------- #
team_lead = Agent(
    model=Gemini(id="gemini-2.0-flash"),
    description="Aggregates stock analysis, company research, and investment strategy.",
    instructions=[
        "Compile stock performance, company analysis, and recommendations.",
        "Ensure all insights are structured in an investor-friendly report.",
        "Rank the top stocks based on combined analysis."
    ],
    markdown=True
)

def get_final_investment_report(symbols):
    market_analysis = get_market_analysis(symbols)
    company_analyses = [get_company_analysis(symbol) for symbol in symbols]
    stock_recommendations = get_stock_recommendations(symbols)

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
    img = Image.open(img_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()
 
logo_base64 = get_base64_logo("static/Hoonartek-V25-White-Color.png")  # Update path as needed

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

        [data-testid="stSidebar"] button {{
            background-color: transparent !important;
            color: grey !important;
            font-weight: bold;
            border: 2px solid grey !important;
            border-radius: 8px; 
            padding: 0.6em 1em;
            margin-top: 10px;
            transition: none !important;  /* Disabling transition for hover */
        }}

        [data-testid="stSidebar"] button:hover {{
        background-color: transparent !important;  /* Remove hover background */
        color: light blue !important;  /* Ensure text color remains the same */
        border: 2px light blue !important;  /* Ensure border stays the same */
        }}

        [data-testid="stSidebar"] h2 {{
            color: white;
            font-size: 1.4rem;
            margin-bottom: 0.5em;
        }}

        [data-testid="stSidebar"] p {{
            color: #f0f8ff;
            font-size: 0.95rem;
            line-height: 1.5;
        }}
    </style>
""", unsafe_allow_html=True)

st.sidebar.markdown(
    f"""
    <div style="text-align: center;">
        <img src="data:image/png;base64,{logo_base64}" style="width: 220px; border-radius: 0px; margin-bottom: 20px;" />
    </div>
    """,
    unsafe_allow_html=True
)


# -- 🧠 Use Case Description
st.sidebar.title("Use Case Details")
st.sidebar.markdown(
    """
AI Stock Report Generator provides real-time analysis of U.S. and Indian stock markets. Users enter stock symbols (e.g., AAPL, MRF.NS) to receive AI-curated reports featuring key metrics, price trends, and performance comparisons. It offers a quick, intuitive way to evaluate and compare multiple stocks."""
)

st.sidebar.subheader("Model Name:\nGemini-2.0-flash")

# --- Stock Symbol Input ---
input_symbols = st.sidebar.text_input("Enter Stock Symbols")
 

# Parse the stock symbols input
stocks_symbols = [symbol.strip() for symbol in input_symbols.split(",")]

st.sidebar.markdown("E.g. AAPL, NVDA (U.S.) or MRF.NS, RELIANCE.NS (India)")

# Generate Investment Report button
if st.sidebar.button("Generate Report"):
    if not stocks_symbols:
        st.sidebar.warning("Please enter at least one stock symbol.")
    else:
        # Generate the final report
        report = get_final_investment_report(stocks_symbols)

        # Display the report
        st.subheader("Investment Report")
        st.markdown(report)

        st.info("This report provides detailed insights, including market performance, company analysis, and investment recommendations.")

        # Interactive Stock Performance Chart
        st.markdown("### 📊 Stock Performance (6-Months)")
        stock_data = yf.download(stocks_symbols, period="6mo")['Close']

        fig = go.Figure()
        for symbol in stocks_symbols:
            fig.add_trace(go.Scatter(x=stock_data.index, y=stock_data[symbol], mode='lines', name=symbol))

        fig.update_layout(title="Stock Performance Over the Last 6 Months",
                          xaxis_title="Date",
                          yaxis_title="Price (in USD)",
                          template="plotly_dark")
        st.plotly_chart(fig)