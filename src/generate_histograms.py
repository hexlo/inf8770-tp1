"""
Génération d'histogrammes pour les métriques de compression Huffman
"""

import matplotlib.pyplot as plt
import numpy as np
from huffman_coding import huffman
import os
import glob


class CompressionAnalyzer:
    """
    Classe pour analyser et visualiser les métriques de compression
    """
    
    def __init__(self):
        self.results = []
    
    def analyze_images(self, image_paths):
        """
        Analyse plusieurs images et collecte les métriques
        """
        for image_path in image_paths:
            if not os.path.exists(image_path):
                print(f"Avertissement: '{image_path}' n'existe pas, ignoré.")
                continue
            
            try:
                print(f"Analyse de: {image_path}")
                metrics = huffman(image_path)
                
                result = {
                    'filename': os.path.basename(image_path),
                    'original_size': metrics['taille_originale'],
                    'compressed_size': metrics['taille_compressee'],
                    'compression_ratio': metrics['ratio_compression'],
                    'compression_percentage': metrics['pourcentage_reduction']
                }
                
                self.results.append(result)
                print(f"  ✓ Traité avec succès\n")
                
                # NOUVEAU: Calculer la distribution des symboles
                try:
                    from PIL import Image
                    img = Image.open(image_path)
                    # Convertir en array numpy et aplatir
                    data = np.array(img).flatten()
                    
                    # Convertir en int si nécessaire pour éviter les problèmes de type
                    if np.issubdtype(data.dtype, np.integer):
                         data = data.astype(int)
                         
                    # Compter les occurrences (0-255)
                    # On utilise np.bincount car c'est plus rapide pour des entiers non-négatifs
                    # minlength=256 assure qu'on a bien tous les symboles possibles
                    counts = np.bincount(data, minlength=256)
                    
                    # Stocker la distribution pour l'histogramme
                    # Utiliser le dossier output absolu
                    script_dir = os.path.dirname(os.path.abspath(__file__))
                    output_dir = os.path.join(os.path.dirname(script_dir), "output")
                    
                    self._create_symbol_distribution_histogram(os.path.basename(image_path), counts, output_dir=output_dir)
                    
                except Exception as e:
                    print(f"  Warning: Impossible de calculer la distribution des symboles: {e}")

                
            except Exception as e:
                print(f"Erreur lors de l'analyse de {image_path}: {e}")
    
    def _create_symbol_distribution_histogram(self, filename, counts, output_dir):
        """
        Crée un histogramme de la distribution des symboles pour une image donnée
        """
        # Créer le répertoire de sortie s'il n'existe pas
        # Note: ceci est redondant si generate_histograms est appelé, mais utile si appelé directement
        if output_dir is None:
             script_dir = os.path.dirname(os.path.abspath(__file__))
             output_dir = os.path.join(os.path.dirname(script_dir), "output")
        os.makedirs(output_dir, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Symboles 0-255
        x = np.arange(256)
        
        # Utiliser une échelle log si la distribution est très inégale (optionnel, ici linéaire par défaut)
        # ax.bar(x, counts, color='#3498db', width=1.0, alpha=0.7)
        ax.fill_between(x, 0, counts, color='#3498db', alpha=0.4)
        ax.plot(x, counts, color='#2980b9', linewidth=1)
        
        ax.set_xlim(0, 255)
        ax.set_xlabel('Valeur du symbole (0-255)', fontsize=10)
        ax.set_ylabel('Fréquence (nombre de pixels)', fontsize=10)
        ax.set_title(f'Distribution des Symboles: {filename}', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Ajouter le nombre total de symboles différents (valeurs > 0)
        num_symbols = np.count_nonzero(counts)
        ax.text(0.95, 0.95, f'Symboles uniques: {num_symbols}', 
                transform=ax.transAxes, ha='right', va='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        output_filename = f"distribution_{os.path.splitext(filename)[0]}.png"
        plt.savefig(os.path.join(output_dir, output_filename), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Distribution sauvegardée: {output_filename}")


    def generate_histograms(self, output_dir=None):
        """
        Génère des histogrammes pour les différentes métriques
        """
        if output_dir is None:
            # Par défaut: dossier 'output' à la racine du projet
            script_dir = os.path.dirname(os.path.abspath(__file__))
            output_dir = os.path.join(os.path.dirname(script_dir), "output")

        if not self.results:
            print("Aucun résultat à visualiser.")
            return
        
        # Créer le répertoire de sortie s'il n'existe pas
        os.makedirs(output_dir, exist_ok=True)
        
        # Configurer le style pour de meilleurs graphiques
        plt.style.use('seaborn-v0_8-darkgrid')
        
        # Extraire les données
        filenames = [r['filename'] for r in self.results]
        original_sizes = [r['original_size'] / 1024 for r in self.results]  # Convertir en Ko
        compressed_sizes = [r['compressed_size'] / 1024 for r in self.results]  # Convertir en Ko
        compression_ratios = [r['compression_ratio'] for r in self.results]
        compression_percentages = [r['compression_percentage'] for r in self.results]
        
        # Histogramme 1: Tailles originales vs compressées
        self._create_size_comparison_histogram(
            filenames, original_sizes, compressed_sizes, output_dir
        )
        
        # Histogramme 2: Ratios de compression
        self._create_ratio_histogram(
            filenames, compression_ratios, output_dir
        )
        
        # Histogramme 3: Pourcentages de réduction
        self._create_percentage_histogram(
            filenames, compression_percentages, output_dir
        )
        
        # Graphique combiné
        self._create_combined_chart(
            filenames, original_sizes, compressed_sizes, 
            compression_ratios, output_dir
        )
        
        print(f"\nHistogrammes sauvegardés dans le répertoire: {output_dir}")
    
    def _create_size_comparison_histogram(self, filenames, original_sizes, 
                                         compressed_sizes, output_dir):
        """
        Crée un histogramme comparant les tailles originales et compressées
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(filenames))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, original_sizes, width, label='Taille originale',
                       color='#FF6B6B', alpha=0.8)
        bars2 = ax.bar(x + width/2, compressed_sizes, width, label='Taille compressée',
                       color='#4ECDC4', alpha=0.8)
        
        ax.set_xlabel('Images', fontsize=12, fontweight='bold')
        ax.set_ylabel('Taille (Ko)', fontsize=12, fontweight='bold')
        ax.set_title('Comparaison des Tailles: Originale vs Compressée', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(filenames, rotation=45, ha='right')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Ajouter les valeurs sur les barres
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}',
                       ha='center', va='bottom', fontsize=8)
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/comparaison_tailles.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_ratio_histogram(self, filenames, ratios, output_dir):
        """
        Crée un histogramme des ratios de compression
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(filenames))
        bars = ax.bar(x, ratios, color='#95E1D3', alpha=0.8, edgecolor='#2C3E50')
        
        ax.set_xlabel('Images', fontsize=12, fontweight='bold')
        ax.set_ylabel('Ratio de Compression', fontsize=12, fontweight='bold')
        ax.set_title('Ratio de Compression par Image', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(filenames, rotation=45, ha='right')
        ax.axhline(y=1, color='red', linestyle='--', linewidth=2, 
                  label='Ratio = 1 (pas de compression)', alpha=0.6)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}x',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/ratio_compression.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_percentage_histogram(self, filenames, percentages, output_dir):
        """
        Crée un histogramme des pourcentages de réduction
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(filenames))
        colors = ['#FF6B6B' if p < 0 else '#51CF66' for p in percentages]
        bars = ax.bar(x, percentages, color=colors, alpha=0.8, edgecolor='#2C3E50')
        
        ax.set_xlabel('Images', fontsize=12, fontweight='bold')
        ax.set_ylabel('Réduction (%)', fontsize=12, fontweight='bold')
        ax.set_title('Pourcentage de Réduction de Taille', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(filenames, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Ajouter les valeurs sur les barres
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom' if height >= 0 else 'top', 
                   fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/pourcentage_reduction.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_combined_chart(self, filenames, original_sizes, compressed_sizes,
                              ratios, output_dir):
        """
        Crée un graphique combiné avec plusieurs sous-graphiques
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        x = np.arange(len(filenames))
        
        # Sous-graphique 1: Tailles en barres groupées
        width = 0.35
        ax1.bar(x - width/2, original_sizes, width, label='Original', 
               color='#FF6B6B', alpha=0.8)
        ax1.bar(x + width/2, compressed_sizes, width, label='Compressé', 
               color='#4ECDC4', alpha=0.8)
        ax1.set_xlabel('Images', fontweight='bold')
        ax1.set_ylabel('Taille (Ko)', fontweight='bold')
        ax1.set_title('Comparaison des Tailles', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(filenames, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Sous-graphique 2: Ratios de compression
        ax2.bar(x, ratios, color='#95E1D3', alpha=0.8, edgecolor='#2C3E50')
        ax2.set_xlabel('Images', fontweight='bold')
        ax2.set_ylabel('Ratio', fontweight='bold')
        ax2.set_title('Ratio de Compression', fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(filenames, rotation=45, ha='right')
        ax2.axhline(y=1, color='red', linestyle='--', alpha=0.6)
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Sous-graphique 3: Économie d'espace (ou augmentation)
        savings = [orig - comp for orig, comp in zip(original_sizes, compressed_sizes)]
        colors = ['#51CF66' if s > 0 else '#FF6B6B' for s in savings]
        ax3.bar(x, savings, color=colors, alpha=0.8, edgecolor='#2C3E50')
        ax3.set_xlabel('Images', fontweight='bold')
        ax3.set_ylabel('Économie (Ko)', fontweight='bold')
        ax3.set_title('Économie d\'Espace', fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(filenames, rotation=45, ha='right')
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Sous-graphique 4: Tableau récapitulatif
        ax4.axis('tight')
        ax4.axis('off')
        
        table_data = []
        for i, filename in enumerate(filenames):
            table_data.append([
                filename,
                f'{original_sizes[i]:.1f}',
                f'{compressed_sizes[i]:.1f}',
                f'{ratios[i]:.2f}x'
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Fichier', 'Original (Ko)', 'Compressé (Ko)', 'Ratio'],
                         cellLoc='center',
                         loc='center',
                         colWidths=[0.4, 0.2, 0.2, 0.2])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 2)
        
        # Styliser l'en-tête
        for i in range(4):
            table[(0, i)].set_facecolor('#2C3E50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        plt.suptitle('Analyse Complète de la Compression Huffman', 
                    fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/analyse_complete.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_summary(self):
        """
        Affiche un résumé textuel des résultats
        """
        if not self.results:
            print("Aucun résultat disponible.")
            return
        
        print("\n" + "="*70)
        print("RÉSUMÉ DE L'ANALYSE DE COMPRESSION")
        print("="*70)
        
        for result in self.results:
            print(f"\nFichier: {result['filename']}")
            print(f"  Taille originale:      {result['original_size']:,} octets")
            print(f"  Taille compressée:     {result['compressed_size']:,} octets")
            print(f"  Ratio de compression:  {result['compression_ratio']:.2f}x")
            print(f"  Réduction:             {result['compression_percentage']:.2f}%")
        
        # Calculer les moyennes
        avg_ratio = np.mean([r['compression_ratio'] for r in self.results])
        avg_percentage = np.mean([r['compression_percentage'] for r in self.results])
        
        print("\n" + "-"*70)
        print("MOYENNES")
        print("-"*70)
        print(f"  Ratio moyen:           {avg_ratio:.2f}x")
        print(f"  Réduction moyenne:     {avg_percentage:.2f}%")
        print("="*70 + "\n")


def generate_compression_histograms(image_paths):
    """
    Génère les histogrammes de compression pour les images spécifiées.
    
    Args:
        image_paths: Liste de chemins d'images ou chemin unique (string)
    """
    # Permettre un seul chemin ou une liste de chemins
    if isinstance(image_paths, str):
        image_paths = [image_paths]
    
    analyzer = CompressionAnalyzer()
    
    if not image_paths:
        print("Aucune image fournie pour l'analyse")
        return
    
    # Analyser les images
    print("Début de l'analyse des images...\n")
    analyzer.analyze_images(image_paths)
    
    # Afficher le résumé
    analyzer.print_summary()
    
    # Générer les histogrammes
    print("Génération des histogrammes...")
    analyzer.generate_histograms()
    
    print("\nAnalyse terminée avec succès!")


def main():
    """
    Point d'entrée CLI - trouve toutes les images et génère les histogrammes
    """
    # Trouver toutes les images PNG dans le répertoire images/
    script_dir = os.path.dirname(os.path.abspath(__file__))
    images_dir = os.path.join(os.path.dirname(script_dir), "images")
    image_paths = glob.glob(os.path.join(images_dir, "*.png"))
    
    if not image_paths:
        print("Aucune image trouvée pour l'analyse dans le répertoire images/")
        print("Veuillez ajouter des images PNG dans le répertoire images/")
        return
    
    generate_compression_histograms(image_paths)


if __name__ == "__main__":
    main()
