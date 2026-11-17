
import os
import joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date

# =====================================================
# 1. PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Sales Analysis and Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "sales_data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "sales_model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "sales_features.pkl")

# =====================================================
# 2. PRODUCT DATABASE
# =====================================================

PRODUCTS = {
    "Laptop": ("Electronics", 55000),
    "Mobile": ("Electronics", 25000),
    "Headphones": ("Electronics", 2000),
    "Tablet": ("Electronics", 18000),
    "Keyboard": ("Electronics", 1200),
    "Mouse": ("Electronics", 700),
    "Monitor": ("Electronics", 12000),
    "Printer": ("Electronics", 9000),

    "T-Shirt": ("Clothing", 600),
    "Shirt": ("Clothing", 1000),
    "Jeans": ("Clothing", 1800),
    "Jacket": ("Clothing", 3000),
    "Dress": ("Clothing", 2200),
    "Hoodie": ("Clothing", 1500),

    "Sneakers": ("Footwear", 2500),
    "Formal Shoes": ("Footwear", 3000),
    "Sandals": ("Footwear", 800),
    "Sports Shoes": ("Footwear", 2800),

    "Mixer": ("Home & Kitchen", 3500),
    "Microwave": ("Home & Kitchen", 12000),
    "Electric Kettle": ("Home & Kitchen", 1500),
    "Cookware Set": ("Home & Kitchen", 2500),
    "Office Chair": ("Home & Kitchen", 6500),
    "Table Lamp": ("Home & Kitchen", 1200),

    "Rice": ("Groceries", 1200),
    "Cooking Oil": ("Groceries", 180),
    "Tea": ("Groceries", 350),
    "Coffee": ("Groceries", 450),
    "Biscuits": ("Groceries", 50),

    "Shampoo": ("Beauty & Personal Care", 300),
    "Face Wash": ("Beauty & Personal Care", 250),
    "Moisturizer": ("Beauty & Personal Care", 500),
    "Perfume": ("Beauty & Personal Care", 1500),

    "Yoga Mat": ("Sports & Fitness", 800),
    "Dumbbells": ("Sports & Fitness", 1800),
    "Football": ("Sports & Fitness", 700),

    "Notebook": ("Stationery", 80),
    "Pen Set": ("Stationery", 150),

    "Board Game": ("Toys & Games", 900),
    "Car Charger": ("Automotive", 600)
}

CATEGORIES = sorted({
    item[0] for item in PRODUCTS.values()
})

REGIONS = [
    "North", "South", "East", "West", "Central"
]

# =====================================================
# 3. CUSTOM CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background-color: #F3F5F8;
    color: #26364A;
    font-family: Arial, sans-serif;
}

.block-container {
    max-width: 1250px;
    padding-top: 1.3rem;
    padding-bottom: 2rem;
    padding-left: 2rem;
    padding-right: 2rem;
}

h1, h2, h3 {
    color: #203653 !important;
    font-weight: 700 !important;
}

p, label, li {
    color: #46566B;
}

/* HERO SECTION */

.hero {
    background:
        linear-gradient(
            90deg,
            rgba(15,35,60,0.88),
            rgba(15,35,60,0.35)
        ),
        url("https://images.unsplash.com/photo-1460925895917-afdab827c52f?w=1600");

    background-size: cover;
    background-position: center;
    border-radius: 16px;
    padding: 42px 35px;
    margin-bottom: 22px;
    min-height: 190px;

    display: flex;
    flex-direction: column;
    justify-content: center;
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 30px !important;
    margin-bottom: 8px;
}

.hero p {
    color: #F1F5F9 !important;
    font-size: 15px;
    margin: 0;
}

/* METRIC CARDS */

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E0E6ED;
    border-radius: 12px;
    padding: 17px;
    box-shadow: 0 2px 7px rgba(30,50,75,0.04);
}

div[data-testid="stMetricLabel"] {
    color: #52647A !important;
}

div[data-testid="stMetricValue"] {
    color: #203653 !important;
    font-weight: 700;
}

/* NAVIGATION CONTAINER */

div[data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #DCE3EB;
    border-radius: 14px;
    background: #FFFFFF;
}

/* PRIMARY BUTTONS */

.stApp div[data-testid="stButton"] button,
.stApp div[data-testid="stDownloadButton"] button,
.stApp div[data-testid="stFormSubmitButton"] button {

    background: #244A73 !important;
    color: #FFFFFF !important;
    border: 1px solid #244A73 !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    min-height: 42px;
}

.stApp div[data-testid="stButton"] button p,
.stApp div[data-testid="stDownloadButton"] button p,
.stApp div[data-testid="stFormSubmitButton"] button p,
.stApp div[data-testid="stButton"] button span,
.stApp div[data-testid="stDownloadButton"] button span,
.stApp div[data-testid="stFormSubmitButton"] button span {
    color: #FFFFFF !important;
}

.stApp div[data-testid="stButton"] button:hover,
.stApp div[data-testid="stDownloadButton"] button:hover,
.stApp div[data-testid="stFormSubmitButton"] button:hover {
    background: #193A60 !important;
    color: #FFFFFF !important;
    border-color: #193A60 !important;
}

/* SECONDARY NAVIGATION BUTTONS */

.stApp div[data-testid="stButton"] button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #244A73 !important;
    border: 1px solid #D7E0E9 !important;
}

.stApp div[data-testid="stButton"] button[kind="secondary"] p,
.stApp div[data-testid="stButton"] button[kind="secondary"] span {
    color: #244A73 !important;
}

/* INPUT FIELDS */

.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stTextArea textarea {
    background-color: #EAF1F7 !important;
    color: #263B50 !important;
    border: 1px solid #C9D7E5 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] > div {
    background-color: #EAF1F7 !important;
    color: #263B50 !important;
    border-color: #C9D7E5 !important;
    border-radius: 8px !important;
}

div[data-baseweb="select"] span {
    color: #263B50 !important;
}

/* FILE UPLOADER */


/* FILE UPLOADER */

section[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border: 1px dashed #9FB2C7 !important;
    border-radius: 12px !important;
    padding: 12px !important;
}

section[data-testid="stFileUploader"] button {
    background: #EAF1F7 !important;
    color: #263B50 !important;
    border: 1px solid #C9D7E5 !important;
    border-radius: 8px !important;
}

section[data-testid="stFileUploader"] button * {
    color: #263B50 !important;
    fill: #263B50 !important;
}


/* SUCCESS AND INFORMATION ALERTS */

div[data-testid="stAlert"] {
    background-color: #E8F5E9;
    border: 1px solid #B7DDBB;
    border-radius: 9px;
}

div[data-testid="stAlert"] p {
    color: #205B32 !important;
}

/* DATA TABLE BORDER */

div[data-testid="stDataFrame"],
div[data-testid="stDataEditor"] {
    border: 1px solid #DCE3EB;
    border-radius: 10px;
}

hr {
    border-color: #DCE3EB;
}

/* MOBILE RESPONSIVE */

@media (max-width: 768px) {

    .block-container {
        padding: 1rem 0.7rem 2rem 0.7rem;
    }

    .hero {
        padding: 25px 20px;
        min-height: 160px;
    }

    .hero h1 {
        font-size: 22px !important;
    }

    .hero p {
        font-size: 13px;
    }

    h2 {
        font-size: 21px !important;
    }

    h3 {
        font-size: 18px !important;
    }

    div[data-testid="stMetric"] {
        padding: 10px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 19px;
    }
}


/* SUCCESS MESSAGE - LIGHT GREEN */

.success-box {
    background-color: #E8F5E9 !important;
    color: #205B32 !important;
    border: 1px solid #B7DDBB;
    border-radius: 10px;
    padding: 15px 20px;
    margin: 10px 0;
    font-weight: 600;
}

/* INFO MESSAGE - LIGHT BLUE */

.info-box {
    background-color: #EAF1F7 !important;
    color: #244A73 !important;
    border: 1px solid #C9D7E5;
    border-radius: 10px;
    padding: 15px 20px;
    margin: 10px 0;
    font-weight: 600;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.stApp {
    border-top: none !important;
}

</style>
""", unsafe_allow_html=True)

# =====================================================
# 4. DATA PREPARATION
# =====================================================

def prepare_data(data):

    data = data.copy()

    aliases = {
        "date": "Date",
        "product": "Product",
        "product name": "Product",
        "category": "Category",
        "quantity": "Quantity",
        "qty": "Quantity",
        "price": "Price",
        "unit price": "Price",
        "discount": "Discount",
        "region": "Region"
    }

    rename_columns = {}

    for column in data.columns:
        clean_name = str(column).strip().lower()

        if clean_name in aliases:
            rename_columns[column] = aliases[clean_name]

    data = data.rename(columns=rename_columns)

    required = ["Product", "Quantity", "Price"]

    missing = [
        col for col in required
        if col not in data.columns
    ]

    if missing:
        raise ValueError(
            "Required columns missing: "
            + ", ".join(missing)
            + ". Please include Product, Quantity and Price."
        )

    if "Date" not in data.columns:
        data["Date"] = pd.NaT

    if "Category" not in data.columns:
        data["Category"] = "Uncategorized"

    if "Discount" not in data.columns:
        data["Discount"] = 0

    if "Region" not in data.columns:
        data["Region"] = "Unknown"

    data["Date"] = pd.to_datetime(
        data["Date"],
        errors="coerce"
    )

    for col in ["Quantity", "Price", "Discount"]:
        data[col] = pd.to_numeric(
            data[col],
            errors="coerce"
        )

    for col in ["Product", "Category", "Region"]:
        data[col] = (
            data[col]
            .fillna("")
            .astype(str)
            .str.strip()
        )

    data = data.dropna(
        subset=["Quantity", "Price", "Discount"]
    )

    data = data[
        (data["Product"] != "")
        & (data["Quantity"] > 0)
        & (data["Price"] >= 0)
        & (data["Discount"] >= 0)
        & (data["Discount"] <= 100)
    ]

    data["Category"] = data["Category"].replace(
        "", "Uncategorized"
    )

    data["Region"] = data["Region"].replace(
        "", "Unknown"
    )

    data = data.drop_duplicates()

    data["Sales"] = (
        data["Quantity"]
        * data["Price"]
        * (1 - data["Discount"] / 100)
    )

    columns = [
        "Date",
        "Product",
        "Category",
        "Quantity",
        "Price",
        "Discount",
        "Region",
        "Sales"
    ]

    return data[columns].reset_index(drop=True)


# =====================================================
# 5. SESSION STATE AND LOAD DATA
# =====================================================

if "page" not in st.session_state:
    st.session_state.page = "Overview"

if "sales_data" not in st.session_state:

    if os.path.exists(DATA_PATH):
        st.session_state.sales_data = pd.read_csv(DATA_PATH)

    else:
        st.session_state.sales_data = pd.DataFrame({
            "Date": ["2025-01-01"],
            "Product": ["Laptop"],
            "Category": ["Electronics"],
            "Quantity": [2],
            "Price": [55000],
            "Discount": [0],
            "Region": ["North"]
        })

try:
    df = prepare_data(st.session_state.sales_data)

except Exception as error:
    st.error(f"Dataset error: {error}")
    st.stop()


# =====================================================
# 6. HERO HEADER
# =====================================================

st.markdown("""
<div class="hero">
<h1>Sales Analysis and Prediction System</h1>
<p>Explore sales performance, understand business trends<br>
and estimate sales using Machine Learning.</p>
</div>
""", unsafe_allow_html=True)


# =====================================================
# 7. TOP NAVIGATION
# =====================================================

pages = [
    "Overview",
    "Data",
    "Analysis",
    "Prediction",
    "About"
]

with st.container(border=True):

    nav_columns = st.columns(5)

    for index, page in enumerate(pages):

        with nav_columns[index]:

            if st.button(
                page,
                key=f"nav_{page}",
                type=(
                    "primary"
                    if st.session_state.page == page
                    else "secondary"
                ),
                use_container_width=True
            ):

                st.session_state.page = page
                st.rerun()

st.write("")


# =====================================================
# 8. OVERVIEW PAGE
# =====================================================

if st.session_state.page == "Overview":

    st.header("Sales Overview")

    if df.empty:

        st.warning("No valid sales records available.")

    else:

        total_sales = df["Sales"].sum()
        total_orders = len(df)
        total_products = df["Product"].nunique()
        total_quantity = df["Quantity"].sum()

        average_order = (
            total_sales / total_orders
            if total_orders else 0
        )

        best_product = (
            df.groupby("Product")["Sales"]
            .sum()
            .idxmax()
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Total Revenue",
            f"₹{total_sales:,.0f}"
        )

        col2.metric(
            "Total Orders",
            f"{total_orders:,}"
        )

        col3.metric(
            "Total Products",
            total_products
        )

        col4, col5, col6 = st.columns(3)

        col4.metric(
            "Quantity Sold",
            f"{total_quantity:,.0f}"
        )

        col5.metric(
            "Average Order Value",
            f"₹{average_order:,.0f}"
        )

        col6.metric(
            "Best Selling Product",
            best_product
        )

        st.divider()

        st.subheader("Recent Sales Records")

        st.dataframe(
            df.tail(5),
            use_container_width=True,
            hide_index=True,
            
        )

        st.markdown(
        '<div class="info-box">Visit the Data page to upload, add, edit or download records.</div>',
         unsafe_allow_html=True
        )


# =====================================================
# 9. DATA MANAGEMENT PAGE
# =====================================================

elif st.session_state.page == "Data":

    st.header("Manage Sales Data")

    st.write(
        "Upload your own CSV dataset or add sales records manually."
    )

    # CSV UPLOAD

    st.subheader("Upload CSV Dataset")

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        key="sales_csv_upload"
    )

    if uploaded_file is not None:

        try:

            uploaded_df = pd.read_csv(uploaded_file)

            cleaned_upload = prepare_data(uploaded_df)

            st.success(
                f"File checked successfully! "
                f"{len(cleaned_upload)} valid records found."
            )

            st.write("Preview:")

            st.dataframe(
                cleaned_upload.head(5),
                use_container_width=True,
                hide_index=True,
                
            )

            if st.button(
                "Use Uploaded Dataset",
                type="primary"
            ):

                st.session_state.sales_data = cleaned_upload.copy()

                st.success(
                    "New dataset loaded successfully!"
                )

                st.rerun()

        except Exception as error:

            st.error(
                f"Unable to upload dataset: {error}"
            )

    st.divider()

    # ADD NEW RECORD

    st.subheader("Add New Sales Record")

    product_options = sorted(PRODUCTS.keys())

    product_choice = st.selectbox(
        "Select Product",
        product_options + ["Other / Custom"],
        key="manual_product"
    )

    if product_choice == "Other / Custom":

        product_name = st.text_input(
            "Enter Custom Product Name"
        )

        category_default = "Uncategorized"
        default_price = 1000.0

    else:

        product_name = product_choice

        category_default, default_price = (
            PRODUCTS[product_choice]
        )

    st.caption(
        f"Suggested category: {category_default} | "
        f"Suggested price: ₹{default_price:,.0f}"
    )

    with st.form("add_sales_record"):

        col1, col2 = st.columns(2)

        with col1:

            category_options = (
                CATEGORIES + ["Uncategorized"]
            )

            category_index = (
                category_options.index(category_default)
                if category_default in category_options
                else 0
            )

            category = st.selectbox(
                "Category",
                category_options,
                index=category_index
            )

            quantity = st.number_input(
                "Quantity",
                min_value=1,
                value=1,
                step=1
            )

            discount = st.selectbox(
                "Discount (%)",
                [0, 5, 10, 15, 20],
                index=0
            )

        with col2:

            price = st.number_input(
                "Price per Product (₹)",
                min_value=0.0,
                value=float(default_price),
                step=100.0
            )

            region = st.selectbox(
                "Region",
                REGIONS,
                index=0
            )

            sale_date = st.date_input(
                "Sale Date",
                value=date.today()
            )

        add_button = st.form_submit_button(
            "Add Record",
            type="primary",
            use_container_width=True
        )

        if add_button:

            if product_name.strip():

                new_record = pd.DataFrame([{
                    "Date": sale_date,
                    "Product": product_name.strip(),
                    "Category": category,
                    "Quantity": quantity,
                    "Price": price,
                    "Discount": discount,
                    "Region": region
                }])

                st.session_state.sales_data = pd.concat(
                    [
                        df.drop(columns=["Sales"]),
                        new_record
                    ],
                    ignore_index=True
                )

                st.success(
                    "Sales record added successfully!"
                )

                st.rerun()

            else:

                st.error(
                    "Please enter a product name."
                )

    st.divider()

    # SEARCH AND FILTER

    st.subheader("Search and Filter Records")

    search_text = st.text_input(
        "Search Product"
    )

    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:

        filter_category = st.selectbox(
            "Filter by Category",
            ["All"] + sorted(
                df["Category"].unique().tolist()
            )
        )

    with filter_col2:

        filter_region = st.selectbox(
            "Filter by Region",
            ["All"] + sorted(
                df["Region"].unique().tolist()
            )
        )

    filtered_data = df.copy()

    if search_text:

        filtered_data = filtered_data[
            filtered_data["Product"].str.contains(
                search_text,
                case=False,
                na=False
            )
        ]

    if filter_category != "All":

        filtered_data = filtered_data[
            filtered_data["Category"] == filter_category
        ]

    if filter_region != "All":

        filtered_data = filtered_data[
            filtered_data["Region"] == filter_region
        ]

    st.write(
        f"Records found: {len(filtered_data):,}"
    )

    # PAGINATION

    page_size = st.selectbox(
        "Rows per page",
        [20, 50, 100],
        index=0
    )

    total_pages = max(
        1,
        int(np.ceil(len(filtered_data) / page_size))
    )

    current_page = st.number_input(
        "Page Number",
        min_value=1,
        max_value=total_pages,
        value=1,
        step=1
    )

    start = (current_page - 1) * page_size
    end = start + page_size

    page_data = filtered_data.iloc[start:end].copy()

    st.caption(
        f"Showing {start + 1 if len(filtered_data) else 0} "
        f"to {min(end, len(filtered_data))} "
        f"of {len(filtered_data)} records"
    )

    # EDIT DATA

    st.subheader("Edit or Delete Records")

    st.caption(
        "Edit cells directly. You can also add or delete rows in this table."
    )

    edited_data = st.data_editor(
        page_data,
        use_container_width=True,
        hide_index=True,
        num_rows="dynamic",
        key="sales_editor",
        height=420,
        
    )

    if st.button(
        "Save Data Changes",
        type="primary"
    ):

        try:

            cleaned_edited = prepare_data(edited_data)

            remaining_data = df.drop(
                index=page_data.index
            )

            updated_data = pd.concat(
                [
                    remaining_data,
                    cleaned_edited
                ],
                ignore_index=True
            )

            st.session_state.sales_data = updated_data

            st.success(
                "Data changes saved successfully!"
            )

            st.rerun()

        except Exception as error:

            st.error(
                f"Unable to save changes: {error}"
            )

    st.divider()

    # DOWNLOAD DATASET

    st.subheader("Download Dataset")

    st.download_button(
        "Download Complete Dataset",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="updated_sales_data.csv",
        mime="text/csv",
        use_container_width=True
    )


# =====================================================
# 10. SALES ANALYSIS PAGE
# =====================================================

elif st.session_state.page == "Analysis":

    st.header("Sales Analysis")

    if df.empty:

        st.warning(
            "No data available for analysis."
        )

    else:

        col1, col2, col3 = st.columns(3)

        with col1:

            selected_product = st.selectbox(
                "Select Product",
                ["All"] + sorted(
                    df["Product"].unique().tolist()
                )
            )

        with col2:

            selected_category = st.selectbox(
                "Select Category",
                ["All"] + sorted(
                    df["Category"].unique().tolist()
                )
            )

        with col3:

            selected_region = st.selectbox(
                "Select Region",
                ["All"] + sorted(
                    df["Region"].unique().tolist()
                )
            )

        filtered_df = df.copy()

        if selected_product != "All":

            filtered_df = filtered_df[
                filtered_df["Product"] == selected_product
            ]

        if selected_category != "All":

            filtered_df = filtered_df[
                filtered_df["Category"] == selected_category
            ]

        if selected_region != "All":

            filtered_df = filtered_df[
                filtered_df["Region"] == selected_region
            ]

        st.write(
            f"Records found: {len(filtered_df):,}"
        )

        st.divider()

        # PRODUCT-WISE HORIZONTAL GRAPH

        st.subheader("Product-wise Sales")

        product_sales = (
            filtered_df.groupby("Product")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        if not product_sales.empty:

            chart_sales = (
                product_sales.head(15)
                .sort_values(ascending=True)
            )

            median_sales = product_sales.median()

            colors = [
                "#269B62" if value >= median_sales
                else "#E05252"
                for value in chart_sales.values
            ]

            fig, ax = plt.subplots(
                figsize=(8, 5)
            )

            ax.barh(
                chart_sales.index,
                chart_sales.values,
                color=colors,
                height=0.65
            )

            ax.set_xlabel("Sales (₹)")
            ax.set_ylabel("Product")

            ax.tick_params(
                axis="y",
                labelsize=9
            )

            ax.grid(
                axis="x",
                alpha=0.2
            )

            ax.set_axisbelow(True)

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

            st.caption(
                "Showing top 15 products | "
                "Green = higher sales | Red = lower sales"
            )

        st.divider()

        # MONTHLY SALES GRAPH

        st.subheader("Monthly Sales Trend")

        dated_data = filtered_df.dropna(
            subset=["Date"]
        )

        if not dated_data.empty:

            monthly_sales = (
                dated_data.groupby(
                    dated_data["Date"].dt.to_period("M")
                )["Sales"]
                .sum()
            )

            monthly_sales.index = (
                monthly_sales.index.astype(str)
            )

            median_monthly = monthly_sales.median()

            fig, ax = plt.subplots(
                figsize=(8, 3.2)
            )

            ax.plot(
                monthly_sales.index,
                monthly_sales.values,
                color="#315D87",
                linewidth=2,
                marker="o"
            )

            for index, value in enumerate(
                monthly_sales.values
            ):

                point_color = (
                    "#269B62"
                    if value >= median_monthly
                    else "#E05252"
                )

                ax.scatter(
                    index,
                    value,
                    color=point_color,
                    s=65,
                    zorder=3
                )

            ax.set_xlabel("Month")
            ax.set_ylabel("Sales (₹)")

            ax.tick_params(
                axis="x",
                rotation=30
            )

            ax.grid(
                True,
                alpha=0.2
            )

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        else:

            st.info(
                "Monthly graph is unavailable because this dataset "
                "does not contain valid dates."
            )

        st.divider()

        # CATEGORY-WISE SALES GRAPH

        st.subheader("Category-wise Sales")

        category_sales = (
            filtered_df.groupby("Category")["Sales"]
            .sum()
            .sort_values(ascending=False)
        )

        if not category_sales.empty:

            median_category = category_sales.median()

            category_colors = [
                "#269B62" if value >= median_category
                else "#E05252"
                for value in category_sales.values
            ]

            fig, ax = plt.subplots(
                figsize=(8, 3.5)
            )

            ax.barh(
                category_sales.index,
                category_sales.values,
                color=category_colors
            )

            ax.set_xlabel("Sales (₹)")
            ax.invert_yaxis()

            ax.grid(
                axis="x",
                alpha=0.2
            )

            plt.tight_layout()

            st.pyplot(
                fig,
                use_container_width=True
            )

            plt.close(fig)

        st.divider()

        # TOP PRODUCTS TABLE

        st.subheader("Top Selling Products")

        top_products = (
            filtered_df.groupby("Product")["Sales"]
            .sum()
            .sort_values(ascending=False)
            .reset_index()
        )

        st.dataframe(
            top_products,
            use_container_width=True,
            hide_index=True,
            
        )

        st.download_button(
            "Download Sales Report",
            data=filtered_df.to_csv(
                index=False
            ).encode("utf-8"),
            file_name="sales_report.csv",
            mime="text/csv"
        )


# =====================================================
# 11. MACHINE LEARNING PREDICTION PAGE
# =====================================================

elif st.session_state.page == "Prediction":

    st.header("Sales Prediction")

    st.write(
        "Estimate sales using your trained Linear Regression model."
    )

    if not os.path.exists(MODEL_PATH):

        st.error(
            "sales_model.pkl not found. Place the trained model "
            "in the same folder as this Python file."
        )

    elif not os.path.exists(FEATURES_PATH):

        st.error(
            "sales_features.pkl not found. Place the feature file "
            "in the same folder as this Python file."
        )

    else:

        try:

            model = joblib.load(MODEL_PATH)

            model_columns = joblib.load(
                FEATURES_PATH
            )

            product_columns = [
                col for col in model_columns
                if col.startswith("Product_")
            ]

            region_columns = [
                col for col in model_columns
                if col.startswith("Region_")
            ]

            trained_products = [
                col.replace("Product_", "", 1)
                for col in product_columns
            ]

            trained_regions = [
                col.replace("Region_", "", 1)
                for col in region_columns
            ]

            available_products = [
                product for product in sorted(trained_products)
                if product in df["Product"].unique()
            ]

            if not available_products:

                st.warning(
                    "The current dataset does not contain products "
                    "known to the trained ML model. Prediction is unavailable."
                )

            elif not trained_regions:

                st.warning(
                    "The saved model does not contain trained region features."
                )

            else:

                st.subheader(
                    "Enter Prediction Details"
                )

                with st.form("ml_prediction_form"):

                    col1, col2 = st.columns(2)

                    with col1:

                        prediction_product = st.selectbox(
                            "Select Product",
                            available_products
                        )

                        if prediction_product in PRODUCTS:

                            default_price = (
                                PRODUCTS[prediction_product][1]
                            )

                        else:

                            product_prices = df.loc[
                                df["Product"] == prediction_product,
                                "Price"
                            ]

                            default_price = float(
                                product_prices.median()
                            )

                        prediction_quantity = st.number_input(
                            "Quantity",
                            min_value=1,
                            value=5
                        )

                        prediction_price = st.number_input(
                            "Price per Product (₹)",
                            min_value=0.0,
                            value=float(default_price),
                            step=100.0
                        )

                    with col2:

                        prediction_discount = st.selectbox(
                            "Discount (%)",
                            [0, 5, 10, 15, 20]
                        )

                        prediction_region = st.selectbox(
                            "Region",
                            trained_regions
                        )

                        prediction_date = st.date_input(
                            "Prediction Date",
                            value=date.today()
                        )

                    predict_button = st.form_submit_button(
                        "Predict Sales",
                        type="primary",
                        use_container_width=True
                    )

                if predict_button:

                    new_data = pd.DataFrame({
                        "Quantity": [prediction_quantity],
                        "Price": [prediction_price],
                        "Discount": [prediction_discount],
                        "Month": [prediction_date.month],
                        "Day": [prediction_date.day],
                        "Product": [prediction_product],
                        "Region": [prediction_region]
                    })

                    new_data = pd.get_dummies(
                        new_data,
                        columns=["Product", "Region"],
                        dtype=int
                    )

                    new_data = new_data.reindex(
                        columns=model_columns,
                        fill_value=0
                    )

                    prediction = model.predict(
                        new_data
                    )[0]

                    st.markdown(
                    f'<div class="success-box">Estimated Sales: ₹{prediction:,.2f}</div>',
                      unsafe_allow_html=True
                    )

                    calculated_sales = (
                        prediction_quantity
                        * prediction_price
                        * (1 - prediction_discount / 100)
                    )

                    st.markdown(
                    f'<div class="info-box">Formula-based Sales: ₹{calculated_sales:,.2f}</div>',
                    unsafe_allow_html=True
                    )

                    st.caption(
                        "ML prediction is an estimate. It may differ "
                        "from formula-based sales."
                    )

        except Exception as error:

            st.error(
                f"Unable to load or use the ML model: {error}"
            )


# =====================================================
# 12. ABOUT PAGE
# =====================================================

elif st.session_state.page == "About":

    st.header("About This Project")

    st.write(
        "Sales Analysis and Prediction System is a Python-based "
        "Data Science project developed to explore sales records, "
        "visualize business performance and demonstrate Machine Learning."
    )

    st.divider()

    st.subheader("Project Objectives")

    st.markdown("""
    - Analyze historical sales records.
    - Identify top-selling products and categories.
    - Understand monthly sales trends.
    - Upload and manage sales datasets.
    - Visualize sales performance through charts.
    - Estimate sales using Machine Learning.
    """)

    st.divider()

    st.subheader("Technologies Used")

    st.markdown("""
    - Python
    - Pandas and NumPy
    - Matplotlib
    - Scikit-learn
    - Streamlit
    - Joblib
    """)

    st.divider()

    st.subheader("Machine Learning Algorithm")

    st.write(
        "The project uses Linear Regression to estimate sales based on "
        "quantity, price, discount, date, product and region."
    )

    st.subheader("Project Limitations")

    st.write(
        "Prediction performance depends on the quality and structure "
        "of the training dataset. The current practice dataset is "
        "synthetic, so it is intended for learning and demonstration."
    )


# =====================================================
# 13. FOOTER
# =====================================================

st.divider()

st.caption(
    "Sales Analysis and Prediction System | "
    "Python, Data Science and Machine Learning"
)
