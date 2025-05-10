import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.ticker as ticker

# styl wykresów
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Wczytanie danych
def load_data():
    orders = pd.read_csv('orders.csv')
    products = pd.read_csv('products.csv')
    orders_products = pd.read_csv('orders_products.csv')
    route_segments = pd.read_csv('route_segments.csv')
    
    # Konwersja czasów 
    if 'segment_start_time' in route_segments.columns and 'segment_end_time' in route_segments.columns:
        route_segments['segment_start_time'] = pd.to_datetime(route_segments['segment_start_time'])
        route_segments['segment_end_time'] = pd.to_datetime(route_segments['segment_end_time'])
    
    return orders, products, orders_products, route_segments

def calculate_actual_delivery_times(orders, route_segments):
    
    # Filtracja STOP 
    stops = route_segments[route_segments['segment_type'] == 'STOP'].copy()
    
    if 'segment_start_time' in stops.columns and 'segment_end_time' in stops.columns:
        #Czas trwania każdego przystanku w sekundach
        stops['actual_delivery_duration'] = (stops['segment_end_time'] - stops['segment_start_time']).dt.total_seconds()
        
        # Łączenie danych z zamówieniami
        delivery_times = stops[['order_id', 'actual_delivery_duration']].dropna()
        return pd.merge(orders, delivery_times, on='order_id', how='inner')
    

def analyze_delivery_times(merged_data):
    
    # Konwersja sekund na minuty
    merged_data['actual_delivery_minutes'] = np.ceil(merged_data['actual_delivery_duration'] / 60)
    merged_data['planned_delivery_minutes'] = np.ceil(merged_data['planned_delivery_duration'] / 60)
    
    # Obliczanie błędu przewidywania w minutach
    merged_data['prediction_error_minutes'] = merged_data['actual_delivery_minutes'] - merged_data['planned_delivery_minutes']
    
    # Histogram rzeczywistego czasu dostawy
    plt.figure(figsize=(12, 6))
    sns.histplot(merged_data['actual_delivery_minutes'], binwidth=1, color='teal', kde=True)
    plt.title('Histogram rzeczywistego czasu dostawy (zaokrąglony do pełnych minut)', fontsize=14)
    plt.xlabel('Czas dostawy (minuty)', fontsize=12)
    plt.ylabel('Liczba dostaw', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('histogram_delivery_time.png', dpi=300)
    plt.close()
    
    # Histogram błędu przewidywania
    plt.figure(figsize=(12, 6))
    sns.histplot(merged_data['prediction_error_minutes'], binwidth=1, color='coral', kde=True)
    plt.title('Histogram błędu przewidywania czasu dostawy', fontsize=14)
    plt.xlabel('Błąd przewidywania (minuty)', fontsize=12)
    plt.ylabel('Liczba dostaw', fontsize=12)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7, label='Idealne przewidywanie')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('histogram_prediction_error.png', dpi=300)
    plt.close()
    
    return merged_data

def analyze_sectors(merged_data):
    """
    Analizuje czasy dostawy w różnych sektorach, aby sprawdzić hipotezę kierowców.
    """
    # Analiza czasu dostawy według sektorów
    sector_data = merged_data.groupby('sector_id').agg({
        'actual_delivery_duration': ['mean', 'median', 'std', 'count'],
        'planned_delivery_duration': ['mean']
    })
    
    sector_data.columns = ['avg_actual_duration', 'median_actual_duration', 'std_actual_duration', 
                          'delivery_count', 'avg_planned_duration']
    
    # Przeliczanie sekund na minuty
    sector_data['avg_actual_duration'] /= 60
    sector_data['median_actual_duration'] /= 60
    sector_data['std_actual_duration'] /= 60
    sector_data['avg_planned_duration'] /= 60
    
    # Obliczenie średniego błędu przewidywania dla każdego sektora
    sector_data['avg_prediction_error'] = sector_data['avg_actual_duration'] - sector_data['avg_planned_duration']
    
    # Sortowanie według rzeczywistego czasu dostawy
    sector_data = sector_data.sort_values('avg_actual_duration', ascending=False).reset_index()
    
    # Wykres porównujący rzeczywisty i planowany czas dostawy dla różnych sektorów
    plt.figure(figsize=(14, 8))
    
    bar_width = 0.35
    x = np.arange(len(sector_data))
    
    plt.bar(x - bar_width/2, sector_data['avg_actual_duration'], bar_width, 
            color='teal', label='Rzeczywisty czas dostawy')
    plt.bar(x + bar_width/2, sector_data['avg_planned_duration'], bar_width, 
            color='coral', label='Planowany czas dostawy')
    
    #  słupki błędów dla odchylenia standardowego
    plt.errorbar(x - bar_width/2, sector_data['avg_actual_duration'], 
                yerr=sector_data['std_actual_duration'], fmt='none', color='black', capsize=5)
    
    plt.title('Porównanie rzeczywistego i planowanego czasu dostawy według sektorów', fontsize=14)
    plt.xlabel('Sektor', fontsize=12)
    plt.ylabel('Średni czas dostawy (minuty)', fontsize=12)
    plt.xticks(x, sector_data['sector_id'].astype(str), rotation=0)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('sector_comparison.png', dpi=300)
    plt.close()
    
    return sector_data
    
def analyze_weight_impact(merged_data, orders_products, products):
   
    # Obliczanie łącznej wagi każdego zamówienia
    order_weights = orders_products.merge(products, on='product_id')
    order_weights['total_weight'] = order_weights['quantity'] * order_weights['weight']
    order_weight_sum = order_weights.groupby('order_id')['total_weight'].sum().reset_index()
    
    # Łączenie z danymi o czasach dostawy
    weight_delivery = merged_data.merge(order_weight_sum, on='order_id')
    
    # Konwersja gramy na kilogramy
    weight_delivery['total_weight_kg'] = weight_delivery['total_weight'] / 1000
    weight_delivery['actual_delivery_minutes'] = weight_delivery['actual_delivery_duration'] / 60
    
    # Kategorie wagowe
    weight_bins = [0, 1, 2, 5, 10, 20, 50, 100, 200, np.inf]
    weight_labels = ['0-1', '1-2', '2-5', '5-10', '10-20', '20-50', '50-100', '100-200', '200+']
    weight_delivery['weight_category'] = pd.cut(weight_delivery['total_weight_kg'], bins=weight_bins, labels=weight_labels)
    
    # Średni czas dostawy według kategorii wagowych
    weight_analysis = weight_delivery.groupby('weight_category').agg({
        'actual_delivery_minutes': ['mean', 'median', 'std', 'count']
    })
    
    weight_analysis.columns = ['avg_delivery_time', 'median_delivery_time', 'std_delivery_time', 'count']
    weight_analysis = weight_analysis.reset_index()
    
    # Wykres zależności czasu dostawy od wagi
    plt.figure(figsize=(14, 8))
    
    plt.bar(weight_analysis['weight_category'], weight_analysis['avg_delivery_time'], 
            color='teal', alpha=0.7)
    
    # Słupki błędów dla odchylenia standardowego
    plt.errorbar(range(len(weight_analysis)), weight_analysis['avg_delivery_time'], 
                yerr=weight_analysis['std_delivery_time'], fmt='none', color='black', capsize=5)
    
    plt.title('Wpływ wagi zamówienia na czas dostawy', fontsize=14)
    plt.xlabel('Waga zamówienia (kg)', fontsize=12)
    plt.ylabel('Średni czas dostawy (minuty)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('weight_impact.png', dpi=300)
    plt.close()
    
    # Scatter plot 
    plt.figure(figsize=(12, 8))
    sns.scatterplot(data=weight_delivery, x='total_weight_kg', y='actual_delivery_minutes', 
                   hue='sector_id', alpha=0.6, palette='viridis')
    
    # Linia trendu
    sns.regplot(data=weight_delivery, x='total_weight_kg', y='actual_delivery_minutes', 
               scatter=False, color='red', line_kws={"linestyle": "--"})
    
    plt.title('Zależność czasu dostawy od wagi zamówienia', fontsize=14)
    plt.xlabel('Waga zamówienia (kg)', fontsize=12)
    plt.ylabel('Rzeczywisty czas dostawy (minuty)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Ograniczenie osi
    plt.xlim(0, min(200, weight_delivery['total_weight_kg'].max() * 1.1))
    plt.ylim(0, min(60, weight_delivery['actual_delivery_minutes'].max() * 1.1))
    
    plt.tight_layout()
    plt.savefig('weight_delivery_scatter.png', dpi=300)
    plt.close()
    
    return weight_analysis

def analyze_driver_performance(merged_data, route_segments):
   
    if 'segment_start_time' in route_segments.columns and 'segment_end_time' in route_segments.columns:
        # Czas trwania każdego segmentu
        route_segments['segment_duration'] = (route_segments['segment_end_time'] - 
                                             route_segments['segment_start_time']).dt.total_seconds() / 60
        
        # Łączymy z danymi o zamówieniach, aby uzyskać sektor
        driver_data = route_segments.merge(merged_data[['order_id', 'sector_id']], on='order_id', how='inner')
        
        # Analizujemy wydajność kierowców według sektorów
        driver_sector = driver_data.groupby(['driver_id', 'sector_id']).agg({
            'segment_duration': ['mean', 'count'],
            'segment_type': 'count'
        }).reset_index()
        
        driver_sector.columns = ['driver_id', 'sector_id', 'avg_segment_duration', 'segment_count', 'total_segments']
        
        # Filtrujemy kierowców z wystarczającą liczbą segmentów
        min_segments = 5
        filtered_driver_data = driver_sector[driver_sector['segment_count'] >= min_segments]
        
        # Wykres pudełkowy wydajności kierowców według sektorów
        plt.figure(figsize=(14, 8))
        sns.boxplot(data=driver_data, x='sector_id', y='segment_duration', hue='segment_type')
        
        plt.title('Rozkład czasów segmentów według sektorów i typów', fontsize=14)
        plt.xlabel('Sektor', fontsize=12)
        plt.ylabel('Czas trwania segmentu (minuty)', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig('driver_sector_performance.png', dpi=300)
        plt.close()
        
        return driver_sector
    else:
        print("Dane o czasie segmentów nie są dostępne dla analizy wydajności kierowców.")
        return None

def analyze_time_of_day(route_segments):
    
    if 'segment_start_time' in route_segments.columns:
        # Dodajemy kolumny z godziną dnia
        route_segments['hour_of_day'] = route_segments['segment_start_time'].dt.hour
        
        # Filtrujemy tylko dostawy (stop) 
        stops = route_segments[route_segments['segment_type'] == 'STOP'].copy()
        
        if 'segment_end_time' in stops.columns:
            # Czas trwania przystanku
            stops['stop_duration'] = (stops['segment_end_time'] - stops['segment_start_time']).dt.total_seconds() / 60
            
            # Analizujemy średni czas dostawy według godziny dnia
            hourly_data = stops.groupby('hour_of_day').agg({
                'stop_duration': ['mean', 'median', 'std', 'count']
            })
            
            hourly_data.columns = ['avg_duration', 'median_duration', 'std_duration', 'count']
            hourly_data = hourly_data.reset_index()
            
            # Wykres zależności czasu dostawy od pory dnia
            plt.figure(figsize=(14, 8))
            
            plt.bar(hourly_data['hour_of_day'], hourly_data['avg_duration'], color='teal', alpha=0.7)
            
            # Słupki błędów dla odchylenia standardowego
            plt.errorbar(hourly_data['hour_of_day'], hourly_data['avg_duration'], 
                        yerr=hourly_data['std_duration'], fmt='none', color='black', capsize=5)
            
            plt.title('Wpływ pory dnia na czas dostawy', fontsize=14)
            plt.xlabel('Godzina dnia', fontsize=12)
            plt.ylabel('Średni czas dostawy (minuty)', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(range(0, 24))
            plt.tight_layout()
            plt.savefig('time_of_day_impact.png', dpi=300)
            plt.close()
            
            return hourly_data
        else:
            print("Dane o czasie zakończenia segmentu nie są dostępne dla analizy wpływu pory dnia.")
            return None
    else:
        print("Dane o czasie rozpoczęcia segmentu nie są dostępne dla analizy wpływu pory dnia.")
        return None

def main():
    # Wczytanie danych
    try:
        orders, products, orders_products, route_segments = load_data()
        print("Dane zostały pomyślnie wczytane.")
        
        # Przygotowanie danych do analizy
        merged_data = calculate_actual_delivery_times(orders, route_segments)
        print("Obliczono rzeczywiste czasy dostaw.")
        
        # Histogramy czasów dostawy i błędów przewidywania
        merged_data = analyze_delivery_times(merged_data)
        print("Wygenerowano histogramy czasów dostawy i błędów przewidywania.")
        
        # Analiza sektorów
        sector_data = analyze_sectors(merged_data)
        print("Przeprowadzono analizę czasu dostawy według sektorów.")
        
        # Analiza wpływu wagi na czas dostawy
        weight_analysis = analyze_weight_impact(merged_data, orders_products, products)
        print("Przeprowadzono analizę wpływu wagi na czas dostawy.")
        
        # Analiza wydajności kierowców
        driver_analysis = analyze_driver_performance(merged_data, route_segments)
        if driver_analysis is not None:
            print("Przeprowadzono analizę wydajności kierowców.")
        
        # Analiza wpływu pory dnia
        time_analysis = analyze_time_of_day(route_segments)
        if time_analysis is not None:
            print("Przeprowadzono analizę wpływu pory dnia na czas dostawy.")
        
        print("\nAnaliza zakończona. Wszystkie wykresy zostały zapisane w bieżącym katalogu.")
        
        # Wyświetlenie kluczowych wyników
        print("\nKluczowe wnioski:")
        
        # Identyfikujemy sektor z najdłuższym czasem dostawy
        problem_sector = sector_data.iloc[0]['sector_id']
        problem_sector_time = sector_data.iloc[0]['avg_actual_duration']
        avg_time = sector_data['avg_actual_duration'].mean()
        
        print(f"1. Sektor {problem_sector} ma najdłuższy średni czas dostawy: {problem_sector_time:.2f} minut, co jest o {(problem_sector_time/avg_time - 1)*100:.1f}% więcej niż średnia dla wszystkich sektorów.")
        
        # Identyfikacja trendów w błędach przewidywania
        avg_error = merged_data['prediction_error_minutes'].mean()
        if avg_error > 0:
            print(f"2. Średni błąd przewidywania wynosi {avg_error:.2f} minut (czasy dostawy są zwykle dłuższe niż planowano).")
        else:
            print(f"2. Średni błąd przewidywania wynosi {avg_error:.2f} minut (czasy dostawy są zwykle krótsze niż planowano).")
        
    except Exception as e:
        print(f"Wystąpił błąd podczas analizy: {str(e)}")

if __name__ == "__main__":
    main()