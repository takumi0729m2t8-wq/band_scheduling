from pulp import LpProblem, LpVariable, LpMaximize, lpSum
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st
import time

VERSION = "1.2.0" #as of 2025-11-11

with open("changelog.md", "r", encoding="utf-8") as f:
    changelog_content = f.read()

st.title("固定枠作成ツール")
st.markdown(f"<p style='text-align: right; color: gray;'>ver. {VERSION}</b></p>", unsafe_allow_html=True)

start_date = datetime(2025, 1, 1)
end_date = datetime(2025, 1, 1)
dates = st.date_input("期間を選択してください", [start_date, end_date])


pre_bands_list = st.text_area("バンド名を改行区切りで入力してください").splitlines()
bands_list = sorted(pre_bands_list)

leftover_bands = []
pre_csv_files = st.file_uploader("各バンドのCSVファイルをアップロードしてください", 
                                accept_multiple_files=True, type=['csv'])
csv_files = sorted(pre_csv_files, key=lambda f: f.name)
per_band_arrays = []


if st.button("実行！"):
    with st.spinner('処理中...'):
        start_date = dates[0]
        end_date = dates[1]
        
        date_list = [int((start_date + timedelta(days=i)).strftime("%m%d")) 
             for i in range((end_date - start_date).days + 1)]
        period_list = [1, 2, 3, 4, 5, 6]
        
        for uploaded_file in csv_files:
            try:
                df = pd.read_csv(uploaded_file, header=None)
                if df.empty:
                    st.error(f"ファイル {uploaded_file.name} が空です")
                    st.stop()
                df = df.dropna(how='all').dropna(axis=1, how='all')
                # (D, P)の形のデータを(P, D)に転置
                data = df.values.astype(np.int8).T
                per_band_arrays.append(data)
            except Exception as e:
                st.error(f"ファイル {uploaded_file.name} の読み込みエラー: {str(e)}")
                st.stop()

        if len(per_band_arrays) != len(bands_list):
            st.error(f"バンド数({len(bands_list)})とCSVファイル数({len(per_band_arrays)})が一致しません")
            st.stop()

        B = len(bands_list)
        D = len(date_list)
        P = len(period_list)
    # P に対する反復用インデックス（0..P-1）
        P_idx = range(P)

        np.set_printoptions(threshold=np.inf)

        #st.write(per_band_arrays)


    # インデックス（反復可能にする）
        B_idx = range(B)          # 0..B-1
        D_idx = range(D)          # 0..D-1
    # 重みの設定
        l = np.zeros((P, D))
        for i in range(6):
            for j in D_idx:
                l[i, j] = (P - abs(2.5 - i)) * (D + j) / (P*D)
        #st.write(l)
    # 3次元の二値変数 x[b,p,d] を辞書で
    # キーはタプル (b,p,d) とする（pは1-based）
        x = LpVariable.dicts("x", [(b, p+1, d) for b in B_idx for p in P_idx for d in D_idx], cat="Binary")
    # 目的関数（全ての x を合計して最大化）
        problem = LpProblem("practice_period", LpMaximize)
        problem += lpSum(l[p, d] * x[b, p+1, d] for b in B_idx for p in P_idx for d in D_idx) - 100*lpSum(3.0 - lpSum(x[(b, p+1, d)] for p in P_idx for d in reversed(D_idx)) for b in B_idx)
    # 制約：各バンドに対して(d, p)の合計が最大3になる
        for b in B_idx:
            problem += lpSum(x[(b, p+1, d)] for p in P_idx for d in reversed(D_idx)) <= 3
        for p in P_idx:
            for d in reversed(D_idx):
                # 各時間枠に1つのバンドのみ
                problem += lpSum(x[(b, p+1, d)] for b in B_idx) <= 1
        for b in B_idx:
            for d in reversed(D_idx):
                # 各バンドは1日に1回のみ
                problem += lpSum(x[(b, p+1, d)] for p in P_idx) <= 1
        # 制約：x[b,p,d]はa[b,p,d]以下
        for b in B_idx:
            for p in P_idx:
                for d in reversed(D_idx):
                    # a は (B, P, D) の順、pは0-based
                    problem += x[(b, p+1, d)] <= per_band_arrays[b][p][d]


            problem.solve()
            # 結果の表示
            #for d in (D_idx):
            #    st.write(f"===== {date_list[d]} =====")
            #    for p in P_idx:
            #        for b in B_idx:
            #            if x[(b, p+1, d)].varValue == 1:
            #                st.write(f"Period {p+1}: {bands_list[b]}")

            def x_dict_to_array(x_dict, B, P, D, p_is_one_based=True):
                arr = np.zeros((B, P, D), dtype=np.int8)
                for key, var in x_dict.items():
                    try:
                        b, p, d = key
                    except Exception:
                        continue
                    val = getattr(var, 'varValue', None)
                    if val is None:
                        v = 0
                    else:
                        v = 1 if float(val) > 0.5 else 0
                    if p_is_one_based:
                        arr[b, p-1, d] = v  # p=1..6 → 0..5
                    else:
                        arr[b, p, d] = v
                return arr

            def per_band_dataframes_from_array(arr, bands_list, date_list, period_list):
                B, P, D = arr.shape
                dfs = {}
                idx = list(period_list)  # 行: 1～6
                cols = list(date_list)   # 列: 日付
                for b, band_label in enumerate(bands_list):
                    # arr[b, :, :] は既に (P, D) の形状
                    df = pd.DataFrame(arr[b, :, :], index=idx, columns=cols)
                    dfs[band_label] = df
                return dfs

            if __name__ == '__main__':
                problem.solve()
                arr = x_dict_to_array(x, B, P, D, p_is_one_based=True)
                dfs_by_band = per_band_dataframes_from_array(arr, bands_list, date_list, period_list)

            #for label, df in dfs_by_band.items():
            #    print(f"--- Band: {label} ---")
            #    print(df)
            #    print()
        

        data = np.full((P,D), "", dtype=object)

        for band, df in dfs_by_band.items():
            if lpSum(df.values.flatten()) == 3:
                    leftover_bands.append(band)
            for i, p in enumerate(df.index):
                for j, d in enumerate(df.columns):
                    if df.at[p, d] == 1:
                        data[i][j] = band
            print(lpSum(df.values.flatten()))

    result_df = pd.DataFrame(data=data, index=period_list, columns=date_list)
    st.write(result_df)
    st.write("固定枠を3枠用意できなかったバンド", leftover_bands)
    

    # ローカルファイルに保存
    result_df.to_csv("固定枠結果.csv", encoding='utf-8_sig', header=True, index=True)

    # ダウンロード用のCSV文字列を生成
    csv_string = result_df.to_csv(encoding='utf-8_sig', header=True, index=True)
    btn = st.download_button(
            label="作成結果をダウンロード",
            data=csv_string.encode('utf-8-sig'),  # バイトデータに変換
            file_name='koteiwaku_data.csv',
            mime='text/csv',
    )

with open("changelog.md", "r", encoding="utf-8") as f:
    changelog_content = f.read()

st.markdown("""
            <style>
            .small-text {
                font-size: 13px;
                line-height: 1.4;
                color: gray;
            }
            </style>
            """, unsafe_allow_html=True)

with st.expander("バージョン履歴", expanded=False):

    st.caption(changelog_content)
