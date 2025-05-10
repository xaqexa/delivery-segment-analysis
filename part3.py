import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime, time
import warnings
import os

# Utworzenie folderu images2, jeśli nie istnieje
os.makedirs('images2', exist_ok=True)

warnings.filterwarnings('ignore')

# Funkcja do wczytywania danych
def load_data():
    orders = pd.read_csv('orders.csv')
    products = pd.read_csv('products.csv')
    orders_products = pd.read_csv('orders_products.csv')
    route_segments = pd.read_csv('route_segments.csv')
    
    return orders, products, orders_products, route_segments

# Funkcja do przetwarzania danych i łączenia tabel
def process_data(orders, products, orders_products, route_segments):
    # Konwersja czasów na obiekty datetime
    for col in ['segment_start_time', 'segment_end_time']:
        route_segments[col] = pd.to_datetime(route_segments[col])
    
    # Obliczanie rzeczywistego czasu segmentu (w sekundach)
    route_segments['segment_duration'] = (route_segments['segment_end_time'] - 
                                          route_segments['segment_start_time']).dt.total_seconds()
    
    # Grupowanie segmentów typu STOP według zamówień i obliczanie łącznego czasu dostawy
    delivery_times = route_segments[route_segments['segment_type'] == 'STOP'].groupby('order_id')['segment_duration'].sum().reset_index()
    delivery_times.rename(columns={'segment_duration': 'actual_delivery_duration'}, inplace=True)
    
    # Łączenie tabel
    df = orders.merge(delivery_times, on='order_id', how='inner')
    
    # Obliczanie wagi całego zamówienia
    # Najpierw łączymy orders_products z products, aby uzyskać wagę każdego produktu
    order_weights = orders_products.merge(products, on='product_id')
    # Następnie mnożymy wagę przez ilość i sumujemy dla każdego zamówienia
    order_weights['total_weight'] = order_weights['weight'] * order_weights['quantity']
    order_weights = order_weights.groupby('order_id')['total_weight'].sum().reset_index()
    
    # Dodajemy informacje o wadze do połączonych danych
    df = df.merge(order_weights, on='order_id', how='left')
    
    # Dodajemy godzinę dostawy z segmentów typu STOP
    delivery_hours = route_segments[route_segments['segment_type'] == 'STOP'].groupby('order_id')['segment_start_time'].min().reset_index()
    delivery_hours['hour_of_day'] = delivery_hours['segment_start_time'].dt.hour
    delivery_hours['is_rush_hour'] = delivery_hours['hour_of_day'].apply(
        lambda x: 1 if (x >= 8 and x <= 10) or (x >= 16 and x <= 18) else 0)
    df = df.merge(delivery_hours[['order_id', 'hour_of_day', 'is_rush_hour']], on='order_id', how='left')
    
    # Obliczanie błędu przewidywania
    df['prediction_error'] = df['actual_delivery_duration'] - df['planned_delivery_duration']
    df['abs_prediction_error'] = abs(df['prediction_error'])
    df['error_percentage'] = (df['prediction_error'] / df['planned_delivery_duration']) * 100
    
    # Kategoryzacja wagi zamówienia
    df['weight_category'] = pd.cut(
        df['total_weight'], 
        bins=[0, 1000, 5000, 20000, 50000, 100000, float('inf')],
        labels=['0-1kg', '1-5kg', '5-20kg', '20-50kg', '50-100kg', '>100kg']
    )
    
    return df

# Funkcja do wizualizacji porównania algorytmów
def visualize_algorithm_comparison(baseline_errors, sector_errors, rf_errors):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    algorithms = ['Baseline (średnia ogólna)', 'Per Sektor', 'Random Forest']
    mae_values = [baseline_errors['mae'], sector_errors['mae'], rf_errors['mae']]
    rmse_values = [baseline_errors['rmse'], sector_errors['rmse'], rf_errors['rmse']]
    
    x = np.arange(len(algorithms))
    width = 0.35
    
    ax.bar(x - width/2, mae_values, width, label='MAE (s)')
    ax.bar(x + width/2, rmse_values, width, label='RMSE (s)')
    
    ax.set_ylabel('Błąd (sekundy)')
    ax.set_title('Porównanie algorytmów przewidywania czasu dostawy')
    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.legend()
    
    # Dodanie wartości na słupkach
    for i, v in enumerate(mae_values):
        ax.text(i - width/2, v + 5, f'{v:.1f}', ha='center')
    
    for i, v in enumerate(rmse_values):
        ax.text(i + width/2, v + 5, f'{v:.1f}', ha='center')
    
    plt.tight_layout()
    plt.savefig('images2/algorithm_comparison.png')
    plt.close()
    
    return fig, ax

# Funkcja do weryfikacji hipotezy o przewidywaniu per sektor
def verify_sector_hypothesis(df):
    print("Weryfikacja hipotezy o przewidywaniu czasów dostawy per sektor")
    
    # 1. Obliczenie średniego czasu dostawy (baseline)
    global_mean = df['actual_delivery_duration'].mean()
    
    # 2. Obliczenie średniego czasu dostawy per sektor
    sector_means = df.groupby('sector_id')['actual_delivery_duration'].mean()
    
    # 3. Obliczenie błędów dla baseline (wszystkie sektory razem)
    baseline_predictions = [global_mean] * len(df)
    baseline_mae = mean_absolute_error(df['actual_delivery_duration'], baseline_predictions)
    baseline_rmse = np.sqrt(mean_squared_error(df['actual_delivery_duration'], baseline_predictions))
    
    # 4. Obliczenie błędów dla przewidywania per sektor
    sector_predictions = df['sector_id'].map(sector_means)
    sector_mae = mean_absolute_error(df['actual_delivery_duration'], sector_predictions)
    sector_rmse = np.sqrt(mean_squared_error(df['actual_delivery_duration'], sector_predictions))
    
    # 5. Podsumowanie wyników
    print(f"Baseline (średnia ogólna): MAE = {baseline_mae:.2f}s, RMSE = {baseline_rmse:.2f}s")
    print(f"Per Sektor: MAE = {sector_mae:.2f}s, RMSE = {sector_rmse:.2f}s")
    print(f"Poprawa MAE: {(baseline_mae - sector_mae):.2f}s ({(baseline_mae - sector_mae) / baseline_mae * 100:.2f}%)")
    
    # 6. Zaawansowana wizualizacja porównawcza per sektor
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='sector_id', y='actual_delivery_duration', data=df)
    plt.axhline(y=global_mean, color='r', linestyle='-', label='Średnia ogólna')
    
    for sector, mean_value in sector_means.items():
        plt.axhline(y=mean_value, color='g', linestyle='--', alpha=0.5)
        plt.text(sector-1, mean_value+20, f'Średnia sektora {sector}: {mean_value:.2f}s', 
                horizontalalignment='center', size='small', color='g', weight='semibold')
    
    plt.title('Rozkład rzeczywistych czasów dostawy według sektorów')
    plt.xlabel('Sektor')
    plt.ylabel('Rzeczywisty czas dostawy (s)')
    plt.legend(['Średnia ogólna', 'Średnie per sektor'])
    plt.tight_layout()
    plt.savefig('images2/sector_hypothesis.png')
    plt.close()
    
    # 7. Wizualizacja poprawy per sektor
    improvements = []
    for sector in df['sector_id'].unique():
        sector_data = df[df['sector_id'] == sector]
        sector_mean = sector_means[sector]
        baseline_error = np.mean(np.abs(sector_data['actual_delivery_duration'] - global_mean))
        sector_error = np.mean(np.abs(sector_data['actual_delivery_duration'] - sector_mean))
        improvement = (baseline_error - sector_error) / baseline_error * 100
        improvements.append({'sector_id': sector, 'improvement': improvement})
    
    improvements_df = pd.DataFrame(improvements)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='sector_id', y='improvement', data=improvements_df)
    plt.axhline(y=0, color='r', linestyle='-')
    plt.title('Procentowa poprawa dokładności predykcji per sektor')
    plt.xlabel('Sektor')
    plt.ylabel('Poprawa (%)')
    plt.tight_layout()
    plt.savefig('images2/sector_improvements.png')
    plt.close()
    
    return {
        'baseline': {'mae': baseline_mae, 'rmse': baseline_rmse},
        'per_sector': {'mae': sector_mae, 'rmse': sector_rmse},
        'improvement': (baseline_mae - sector_mae) / baseline_mae * 100
    }

# Funkcja do tworzenia i ewaluacji zaawansowanego modelu
def create_advanced_model(df):
    print("\nTworzenie zaawansowanego modelu przewidywania czasu dostawy")
    
    # Przygotowanie cech dla modelu
    features = ['sector_id', 'total_weight', 'hour_of_day', 'is_rush_hour']
    X = df[features]
    y = df['actual_delivery_duration']
    
    # Podział na zbiór treningowy i testowy
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Utworzenie i trening modelu Random Forest
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Ewaluacja modelu
    y_pred = rf_model.predict(X_test)
    rf_mae = mean_absolute_error(y_test, y_pred)
    rf_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    rf_r2 = r2_score(y_test, y_pred)
    
    print(f"Random Forest Model: MAE = {rf_mae:.2f}s, RMSE = {rf_rmse:.2f}s, R² = {rf_r2:.2f}")
    
    # Ważność cech
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': rf_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\nWażność cech:")
    for i, row in feature_importance.iterrows():
        print(f"{row['Feature']}: {row['Importance']:.4f}")
    
    # Wizualizacja ważności cech
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Ważność cech w modelu Random Forest')
    plt.tight_layout()
    plt.savefig('images2/feature_importance.png')
    plt.close()
    
    # Wizualizacja rzeczywistych vs. przewidzianych czasów
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
    plt.xlabel('Rzeczywisty czas dostawy (s)')
    plt.ylabel('Przewidywany czas dostawy (s)')
    plt.title('Porównanie rzeczywistych i przewidywanych czasów dostawy')
    plt.tight_layout()
    plt.savefig('images2/prediction_vs_actual.png')
    plt.close()
    
    # Analiza błędów w zależności od wagi zamówienia
    error_by_weight = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'error': np.abs(y_test - y_pred),
        'weight': X_test['total_weight']
    })
    
    # Przedziały wagowe
    error_by_weight['weight_bin'] = pd.cut(
        error_by_weight['weight'], 
        bins=[0, 5000, 10000, 20000, 50000, 100000, float('inf')],
        labels=['0-5kg', '5-10kg', '10-20kg', '20-50kg', '50-100kg', '>100kg']
    )
    
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='weight_bin', y='error', data=error_by_weight)
    plt.title('Błąd predykcji w zależności od wagi zamówienia')
    plt.xlabel('Przedział wagowy')
    plt.ylabel('Bezwzględny błąd predykcji (s)')
    plt.tight_layout()
    plt.savefig('images2/error_by_weight.png')
    plt.close()
    
    # Analiza błędów w zależności od pory dnia
    error_by_hour = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'error': np.abs(y_test - y_pred),
        'hour': X_test['hour_of_day']
    })
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='hour', y='error', data=error_by_hour)
    plt.title('Błąd predykcji w zależności od pory dnia')
    plt.xlabel('Godzina dnia')
    plt.ylabel('Bezwzględny błąd predykcji (s)')
    plt.tight_layout()
    plt.savefig('images2/error_by_hour.png')
    plt.close()
    
    return {
        'model': rf_model,
        'mae': rf_mae,
        'rmse': rf_rmse,
        'r2': rf_r2,
        'feature_importance': feature_importance
    }

# Analiza korelacji między cechami a czasem dostawy
def analyze_correlations(df):
    print("\nAnaliza korelacji między cechami a czasem dostawy")
    
    # Wybór liczbowych cech do analizy korelacji
    numeric_features = ['sector_id', 'total_weight', 'hour_of_day', 'is_rush_hour', 
                        'planned_delivery_duration', 'actual_delivery_duration']
    
    corr_df = df[numeric_features].corr()
    
    # Macierz korelacji
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title('Macierz korelacji cech')
    plt.tight_layout()
    plt.savefig('images2/correlation_matrix.png')
    plt.close()
    
    # Korelacja z czasem dostawy
    corr_with_delivery = corr_df['actual_delivery_duration'].sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=corr_with_delivery.index, y=corr_with_delivery.values)
    plt.title('Korelacja cech z rzeczywistym czasem dostawy')
    plt.xlabel('Cecha')
    plt.ylabel('Współczynnik korelacji')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('images2/correlation_with_delivery.png')
    plt.close()
    
    # Wykresy rozrzutu dla wybranych cech
    for feature in ['total_weight', 'planned_delivery_duration']:
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x=feature, y='actual_delivery_duration', data=df, alpha=0.5)
        plt.title(f'Zależność czasu dostawy od {feature}')
        plt.xlabel(feature)
        plt.ylabel('Rzeczywisty czas dostawy (s)')
        plt.tight_layout()
        plt.savefig(f'images2/scatter_{feature}.png')
        plt.close()
    
    return corr_df

# Funkcja do analizy czasów dostaw według sektorów
def analyze_sectors(df):
    print("\nAnaliza czasów dostaw według sektorów")
    
    # Statystyki per sektor
    sector_stats = df.groupby('sector_id')['actual_delivery_duration'].agg(['mean', 'median', 'std']).reset_index()
    print(sector_stats)
    
    # Wykres średnich czasów dostawy per sektor
    plt.figure(figsize=(12, 6))
    sns.barplot(x='sector_id', y='mean', data=sector_stats)
    plt.title('Średni czas dostawy według sektorów')
    plt.xlabel('Sektor')
    plt.ylabel('Średni czas dostawy (s)')
    plt.tight_layout()
    plt.savefig('images2/sector_mean_times.png')
    plt.close()
    
    # Wykres odchyleń standardowych per sektor
    plt.figure(figsize=(12, 6))
    sns.barplot(x='sector_id', y='std', data=sector_stats)
    plt.title('Zmienność czasu dostawy według sektorów')
    plt.xlabel('Sektor')
    plt.ylabel('Odchylenie standardowe (s)')
    plt.tight_layout()
    plt.savefig('images2/sector_std_times.png')
    plt.close()
    
    # Badanie zależności między sektorem a wagą zamówienia
    sector_weight = df.groupby('sector_id')['total_weight'].mean().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(x='sector_id', y='total_weight', data=sector_weight)
    plt.title('Średnia waga zamówienia według sektorów')
    plt.xlabel('Sektor')
    plt.ylabel('Średnia waga (g)')
    plt.tight_layout()
    plt.savefig('images2/sector_weight.png')
    plt.close()
    
    # Analiza sektorów pod kątem godzin dostawy
    hour_by_sector = pd.crosstab(df['sector_id'], df['hour_of_day'])
    hour_by_sector_norm = hour_by_sector.div(hour_by_sector.sum(axis=1), axis=0)
    
    plt.figure(figsize=(14, 8))
    sns.heatmap(hour_by_sector_norm, cmap='YlGnBu', annot=False)
    plt.title('Rozkład godzin dostawy według sektorów')
    plt.xlabel('Godzina dnia')
    plt.ylabel('Sektor')
    plt.tight_layout()
    plt.savefig('images2/sector_hour_heatmap.png')
    plt.close()
    
    return sector_stats

# Główna funkcja analizy
def main():
    print("Rozpoczęcie analizy danych dostaw...")
    
    # Wczytanie danych
    try:
        orders, products, orders_products, route_segments = load_data()
        print(f"Wczytano dane: {len(orders)} zamówień, {len(products)} produktów, "
              f"{len(orders_products)} powiązań zamówienie-produkt, {len(route_segments)} segmentów trasy")
    except Exception as e:
        print(f"Błąd wczytywania danych: {e}")
        print("Proszę upewnić się, że pliki CSV są dostępne w bieżącym katalogu.")
        return
    
    # Przetwarzanie danych
    df = process_data(orders, products, orders_products, route_segments)
    print(f"Przetworzono dane: {len(df)} zamówień z pełnymi informacjami")
    
    # Podstawowe statystyki
    print("\nPodstawowe statystyki:")
    print(f"Średni planowany czas dostawy: {df['planned_delivery_duration'].mean():.2f}s")
    print(f"Średni rzeczywisty czas dostawy: {df['actual_delivery_duration'].mean():.2f}s")
    print(f"Średni błąd przewidywania: {df['prediction_error'].mean():.2f}s")
    print(f"Średni błąd procentowy: {df['error_percentage'].mean():.2f}%")
    
    # Histogram rzeczywistych czasów dostawy
    plt.figure(figsize=(12, 6))
    plt.hist(df['actual_delivery_duration'], bins=50, alpha=0.7)
    plt.axvline(df['actual_delivery_duration'].mean(), color='r', linestyle='dashed', linewidth=1)
    plt.title('Histogram rzeczywistych czasów dostawy')
    plt.xlabel('Czas dostawy (s)')
    plt.ylabel('Liczba zamówień')
    plt.text(df['actual_delivery_duration'].mean() + 20, plt.ylim()[1] * 0.9, 
             f'Średnia: {df["actual_delivery_duration"].mean():.2f}s')
    plt.tight_layout()
    plt.savefig('images2/delivery_time_histogram.png')
    plt.close()
    
    # Histogram błędów przewidywania
    plt.figure(figsize=(12, 6))
    plt.hist(df['prediction_error'], bins=50, alpha=0.7)
    plt.axvline(0, color='r', linestyle='dashed', linewidth=1)
    plt.title('Histogram błędów przewidywania czasu dostawy')
    plt.xlabel('Błąd przewidywania (s)')
    plt.ylabel('Liczba zamówień')
    plt.tight_layout()
    plt.savefig('images2/prediction_error_histogram.png')
    plt.close()
    
    # Analiza korelacji
    corr_df = analyze_correlations(df)
    
    # Analiza sektorów
    sector_stats = analyze_sectors(df)
    
    # Weryfikacja hipotezy o sektorach
    sector_hypothesis = verify_sector_hypothesis(df)
    
    # Tworzenie zaawansowanego modelu
    advanced_model = create_advanced_model(df)
    
    # Porównanie wszystkich algorytmów
    visualize_algorithm_comparison(
        {'mae': sector_hypothesis['baseline']['mae'], 'rmse': sector_hypothesis['baseline']['rmse']},
        {'mae': sector_hypothesis['per_sector']['mae'], 'rmse': sector_hypothesis['per_sector']['rmse']},
        {'mae': advanced_model['mae'], 'rmse': advanced_model['rmse']}
    )
    
    print("\nAnaliza zakończona! Wszystkie wykresy zostały zapisane w folderze 'images2'.")

if __name__ == "__main__":
    main()