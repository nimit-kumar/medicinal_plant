# app.py
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from flask import Flask, render_template, request, jsonify, url_for
from werkzeug.utils import secure_filename
import pickle
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# Use the correct model path - using the finetuned model
MODEL_PATH = r"C:\Users\nimit\Music\.vscode\medicinal plant\best_finetuned_model_20260219_153441.h5"
CLASS_INDICES_PATH = r"C:\Users\nimit\Music\.vscode\medicinal plant\class_indices_20260219_153441.pkl"

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# ============================================
# COMPREHENSIVE 40+ INDIAN MEDICINAL PLANTS DATABASE
# ============================================
medicinal_plants_database = {
    'tulsi': {
        'common_name': 'Holy Basil (Tulsi)',
        'scientific_name': 'Ocimum sanctum',
        'hindi_name': 'तुलसी',
        'family': 'Lamiaceae',
        'description': 'Tulsi is a sacred plant in Hinduism and is revered as a goddess. It has immense medicinal properties and is used in Ayurveda for thousands of years. Known as the "Queen of Herbs" and "Elixir of Life".',
        'medicinal_uses': [
            'Treats respiratory disorders like asthma, bronchitis, cough, cold',
            'Boosts immunity and prevents infections',
            'Reduces stress, anxiety and promotes mental clarity',
            'Anti-inflammatory for arthritis and joint pain',
            'Antimicrobial and antibacterial properties',
            'Treats fever and common cold effectively',
            'Digestive health and appetite improvement',
            'Skin disorders and wound healing',
            'Malaria and dengue fever treatment',
            'Headache and earache relief'
        ],
        'health_benefits': [
            'Enhances lung function and respiratory health',
            'Improves digestion and metabolism',
            'Protects against bacterial and viral infections',
            'Reduces blood sugar levels naturally',
            'Promotes heart health and circulation',
            'Anti-aging and rejuvenating properties',
            'Liver protective and detoxifying',
            'Adaptogenic - helps body cope with stress',
            'Rich in antioxidants like eugenol',
            'Strengthens nervous system'
        ],
        'how_to_use': [
            'Tulsi tea: Boil 5-6 leaves in water for 5-10 minutes',
            'Chew 2-3 fresh leaves daily on empty stomach',
            'Tulsi juice with honey for cough and cold',
            'Tulsi powder (1 tsp) with warm water',
            'Tulsi oil for skin applications and massage',
            'Gargle with tulsi water for sore throat',
            'Tulsi drops in ears for earache',
            'Tulsi paste on skin for infections',
            'Tulsi kadha for fever and immunity',
            'Dried tulsi leaves in soups and teas'
        ],
        'precautions': [
            'Avoid during pregnancy without doctor consultation',
            'Consult doctor before surgery (may slow blood clotting)',
            'May lower blood sugar - monitor if diabetic',
            'May interact with blood thinning medications',
            'Avoid excessive use during breastfeeding',
            'Start with small doses if new to tulsi',
            'May cause nausea in some individuals',
            'Not recommended for infants under 2 years'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Katu, Tikta (Pungent, Bitter)',
            'Guna (Quality)': 'Laghu, Ruksha (Light, Dry)',
            'Virya (Potency)': 'Ushna (Heating)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Kapha and Vata',
            'Prabhava (Special)': 'Immunomodulator, Adaptogen'
        }
    },
    'neem': {
        'common_name': 'Neem',
        'scientific_name': 'Azadirachta indica',
        'hindi_name': 'नीम',
        'family': 'Meliaceae',
        'description': 'Neem is a tree in the mahogany family. It is considered a versatile medicinal plant and is a key ingredient in many Ayurvedic medicines. Known as the "Village Pharmacy" and "Nature\'s Drugstore".',
        'medicinal_uses': [
            'Treats skin diseases like eczema, psoriasis, acne',
            'Powerful blood purifier and detoxifier',
            'Dental care - treats gum diseases and plaque',
            'Liver disorders and jaundice',
            'Wound healing and antiseptic',
            'Malaria and fever treatment',
            'Diabetes management',
            'Anti-parasitic and insect repellent',
            'Hair problems - dandruff, lice',
            'Eye disorders and conjunctivitis'
        ],
        'health_benefits': [
            'Antibacterial and antiviral properties',
            'Antifungal effects for skin conditions',
            'Anti-inflammatory for joints and skin',
            'Immune modulator and booster',
            'Blood purifier and detoxifier',
            'Dental health and oral hygiene',
            'Liver protective and regenerative',
            'Anti-diabetic properties',
            'Anti-aging and skin rejuvenation',
            'Contraceptive properties (traditional)'
        ],
        'how_to_use': [
            'Neem leaves paste for skin diseases',
            'Neem oil for hair growth and dandruff',
            'Neem twigs for brushing teeth',
            'Neem tea for internal detox',
            'Bath with neem water for skin health',
            'Neem capsules as supplement',
            'Neem powder with water',
            'Neem cream for skin infections',
            'Neem leaf juice for diabetes',
            'Neem soap for acne and skin problems'
        ],
        'precautions': [
            'Avoid during pregnancy and breastfeeding',
            'May affect male fertility in high doses',
            'Reduce dose in children',
            'Consult doctor for long-term use',
            'May lower blood sugar - monitor if diabetic',
            'Avoid if trying to conceive',
            'Not for infants',
            'May cause liver damage in excessive doses'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Tikta (Bitter)',
            'Guna (Quality)': 'Laghu, Ruksha (Light, Dry)',
            'Virya (Potency)': 'Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Pitta and Kapha',
            'Prabhava (Special)': 'Kushthaghna (Anti-skin disorder)'
        }
    },
    'aloevera': {
        'common_name': 'Aloe Vera',
        'scientific_name': 'Aloe barbadensis miller',
        'hindi_name': 'घृतकुमारी',
        'family': 'Asphodelaceae',
        'description': 'Aloe Vera is a succulent plant species known for its medicinal and cosmetic properties. It has been used in Ayurveda for centuries for skin and digestive health. Known as "Ghritkumari" meaning "girl who brings youth".',
        'medicinal_uses': [
            'Heals burns, wounds and skin irritations',
            'Moisturizes skin and treats sunburn',
            'Aids digestion and relieves constipation',
            'Reduces dental plaque and gum inflammation',
            'Treats acne and skin conditions',
            'Anti-inflammatory for arthritis',
            'Immune system booster',
            'Detoxifies body',
            'Hair growth promoter',
            'Reduces blood sugar levels'
        ],
        'health_benefits': [
            'Rich in vitamins A, C, E, B12, folic acid',
            'Contains minerals like calcium, magnesium, zinc',
            'Soothes and heals skin conditions',
            'Supports digestive health',
            'Boosts collagen production',
            'Anti-inflammatory properties',
            'Hydrates and moisturizes',
            'Promotes hair growth',
            'Alkalizes body',
            'Antioxidant properties'
        ],
        'how_to_use': [
            'Apply fresh gel directly on burns and wounds',
            'Drink aloe vera juice (30ml) for digestive health',
            'Use as face mask for acne and skin care',
            'Apply on hair for dandruff and hair growth',
            'Mix with smoothies and juices',
            'Aloe vera gel with turmeric for skin',
            'As after-sun lotion',
            'In homemade face packs',
            'Aloe vera with honey for cough',
            'Aloe vera water for detox'
        ],
        'precautions': [
            'Some people may be allergic - do patch test',
            'Avoid during pregnancy without consultation',
            'May cause diarrhea in large quantities',
            'Do not consume aloe vera latex (yellow sap)',
            'Remove green rind completely before use',
            'Consult doctor if on medications',
            'Start with small doses',
            'May interact with diabetes medications'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Tikta, Madhura (Bitter, Sweet)',
            'Guna (Quality)': 'Guru, Snigdha (Heavy, Unctuous)',
            'Virya (Potency)': 'Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Madhura (Sweet)',
            'Dosha Effect': 'Balances Pitta and Vata',
            'Prabhava (Special)': 'Rasayana (Rejuvenative)'
        }
    },
    'ashwagandha': {
        'common_name': 'Ashwagandha (Winter Cherry)',
        'scientific_name': 'Withania somnifera',
        'hindi_name': 'अश्वगंधा',
        'family': 'Solanaceae',
        'description': 'Ashwagandha is one of the most important herbs in Ayurveda. It is known for its rejuvenating and stress-relieving properties. The name means "smell of horse" referring to its strength-giving properties and horse-like vitality.',
        'medicinal_uses': [
            'Reduces stress and anxiety naturally',
            'Improves energy and stamina',
            'Enhances brain function and memory',
            'Boosts male reproductive health and fertility',
            'Supports immune system',
            'Anti-inflammatory for arthritis',
            'Improves sleep quality and insomnia',
            'Anti-aging and rejuvenation',
            'Thyroid function support',
            'Muscle strength and recovery'
        ],
        'health_benefits': [
            'Adaptogenic - helps body manage stress',
            'Reduces cortisol levels (stress hormone)',
            'Increases muscle strength and endurance',
            'Improves brain function and cognition',
            'Lowers blood sugar levels',
            'Reduces inflammation',
            'Enhances sexual health and libido',
            'Neuroprotective benefits',
            'Cardiovascular health',
            'Anti-cancer properties (research)'
        ],
        'how_to_use': [
            'Ashwagandha powder (1/2 to 1 tsp) with warm milk at night',
            'Ashwagandha capsules (300-500mg) as supplement',
            'Decoction with other herbs',
            'With honey or ghee for better absorption',
            'In Ayurvedic formulations like Chyawanprash',
            'Ashwagandha tea',
            'With smoothies and warm beverages',
            'Ashwagandha root powder with water',
            'Ashwagandha oil for massage',
            'Consult practitioner for dosage'
        ],
        'precautions': [
            'Avoid during pregnancy and breastfeeding',
            'May interact with sedatives and thyroid medication',
            'Consult doctor if taking diabetes medication',
            'May cause stomach upset in some people',
            'Avoid if have autoimmune diseases',
            'Not for hyperthyroidism without supervision',
            'May cause drowsiness - avoid driving',
            'May increase testosterone - caution for hormone-sensitive conditions'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Tikta, Kashaya (Bitter, Astringent)',
            'Guna (Quality)': 'Laghu, Snigdha (Light, Unctuous)',
            'Virya (Potency)': 'Ushna (Heating)',
            'Vipaka (Post-digestive)': 'Madhura (Sweet)',
            'Dosha Effect': 'Balances Vata and Kapha',
            'Prabhava (Special)': 'Balya (Strength promoter), Vajikarana (Aphrodisiac)'
        }
    },
    'amla': {
        'common_name': 'Amla (Indian Gooseberry)',
        'scientific_name': 'Phyllanthus emblica',
        'hindi_name': 'आंवला',
        'family': 'Phyllanthaceae',
        'description': 'Amla is one of the richest natural sources of Vitamin C and a powerful rejuvenator in Ayurveda. It is considered a divine herb for longevity and health. Known as "Dhatri" meaning "mother nurse" due to its nurturing properties.',
        'medicinal_uses': [
            'Boosts immunity and prevents colds',
            'Promotes hair growth and prevents graying',
            'Improves eyesight and eye health',
            'Regulates blood sugar levels',
            'Enhances digestion and metabolism',
            'Lowers cholesterol and blood pressure',
            'Anti-inflammatory for joints',
            'Liver protective and detoxifying',
            'Anti-aging and rejuvenation',
            'Improves memory and brain function'
        ],
        'health_benefits': [
            'Extremely high in Vitamin C (20x orange)',
            'Rich in antioxidants and tannins',
            'Supports liver function and detoxification',
            'Lowers cholesterol and blood pressure',
            'Anti-aging properties for skin and hair',
            'Improves brain function and memory',
            'Strengthens heart and lungs',
            'Enhances iron absorption',
            'Natural blood purifier',
            'Anti-inflammatory effects'
        ],
        'how_to_use': [
            'Eat fresh amla fruit daily for immunity',
            'Amla juice (30ml) with water on empty stomach',
            'Use amla powder in hair oils for growth',
            'Amla murabba as digestive',
            'Amla candies for Vitamin C',
            'In Chyawanprash daily',
            'Amla tea for health',
            'Pickled amla with meals',
            'Dried amla as snack',
            'Amla with honey for cough'
        ],
        'precautions': [
            'May cause acidity in some people',
            'Monitor blood sugar if diabetic',
            'Avoid in case of hyperacidity',
            'Consult doctor if taking blood thinners',
            'Start with small doses',
            'May interact with certain medications',
            'Not for excessive consumption',
            'May cause constipation in some'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Amla, Madhura, Tikta, Kashaya (Sour, Sweet, Bitter, Astringent)',
            'Guna (Quality)': 'Laghu, Ruksha (Light, Dry)',
            'Virya (Potency)': 'Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Madhura (Sweet)',
            'Dosha Effect': 'Balances all three Doshas (Tridoshic)',
            'Prabhava (Special)': 'Rasayana (Rejuvenative), Vayasthapana (Anti-aging)'
        }
    },
    'giloy': {
        'common_name': 'Giloy (Amruta Balli)',
        'scientific_name': 'Tinospora cordifolia',
        'hindi_name': 'गिलोय',
        'family': 'Menispermaceae',
        'description': 'Giloy is a powerful immunomodulator known as "Amrita" in Ayurveda, meaning root of immortality. It is considered a wonder herb for immunity and is known to boost vitality and fight chronic fevers.',
        'medicinal_uses': [
            'Boosts immunity and fights infections',
            'Reduces fever and manages dengue',
            'Improves digestion and treats acidity',
            'Manages diabetes and blood sugar',
            'Reduces stress and anxiety',
            'Treats respiratory disorders',
            'Liver protective and detoxifier',
            'Anti-arthritic properties',
            'Skin diseases and infections',
            'Urinary tract infections'
        ],
        'health_benefits': [
            'Powerful antioxidant properties',
            'Anti-inflammatory effects',
            'Immunomodulator - balances immune system',
            'Liver protective qualities',
            'Anti-diabetic properties',
            'Anti-stress adaptogen',
            'Anti-aging effects',
            'Improves metabolic rate',
            'Purifies blood',
            'Enhances cognitive function'
        ],
        'how_to_use': [
            'Drink giloy juice (20ml) daily on empty stomach',
            'Take giloy powder (1 tsp) with honey or water',
            'Use giloy capsules as supplements',
            'Make giloy kadha for fever and immunity',
            'Giloy stem decoction',
            'Giloy tablets for convenience',
            'With other herbs for synergy',
            'Giloy water for general health',
            'Giloy leaves paste for skin',
            'Giloy root powder for arthritis'
        ],
        'precautions': [
            'May lower blood sugar significantly',
            'Avoid during pregnancy and breastfeeding',
            'Consult doctor if taking diabetes medication',
            'May cause constipation in some people',
            'Avoid in autoimmune diseases',
            'Monitor blood sugar regularly',
            'Start with low doses',
            'May interact with immunosuppressants'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Tikta, Kashaya (Bitter, Astringent)',
            'Guna (Quality)': 'Guru, Snigdha (Heavy, Unctuous)',
            'Virya (Potency)': 'Ushna (Heating)',
            'Vipaka (Post-digestive)': 'Madhura (Sweet)',
            'Dosha Effect': 'Balances all three Doshas',
            'Prabhava (Special)': 'Rasayana (Rejuvenative), Jwaraghna (Anti-pyretic)'
        }
    },
    'brahmi': {
        'common_name': 'Brahmi (Water Hyssop)',
        'scientific_name': 'Bacopa monnieri',
        'hindi_name': 'ब्राह्मी',
        'family': 'Plantaginaceae',
        'description': 'Brahmi is renowned for enhancing brain function, memory, and cognitive abilities. It is considered a "Medhya Rasayana" (brain rejuvenator) in Ayurveda. The name comes from "Brahma" - the creator god, indicating its ability to enhance consciousness.',
        'medicinal_uses': [
            'Improves memory and learning ability',
            'Reduces anxiety and stress',
            'Enhances concentration and focus',
            'Treats epilepsy and seizures',
            'Promotes hair growth and health',
            'Anti-inflammatory for brain',
            'Neuroprotective effects',
            'Mental clarity and alertness',
            'ADHD management',
            'Insomnia and sleep disorders'
        ],
        'health_benefits': [
            'Neuroprotective properties for brain health',
            'Antioxidant effects protect brain cells',
            'Improves blood circulation to brain',
            'Calms nervous system naturally',
            'Anti-inflammatory properties',
            'Enhances cognitive function',
            'Reduces ADHD symptoms',
            'Anti-aging for brain',
            'Improves synaptic communication',
            'Reduces beta-amyloid plaques'
        ],
        'how_to_use': [
            'Take brahmi powder (1/2 tsp) with ghee or honey',
            'Use brahmi oil for head massage',
            'Drink brahmi tea for mental clarity',
            'Apply brahmi paste on hair for growth',
            'Brahmi capsules as supplement',
            'Brahmi ghee for brain health',
            'With milk before bed',
            'Brahmi juice for memory',
            'Brahmi leaves in salads',
            'Brahmi medicated oil for hair'
        ],
        'precautions': [
            'May cause digestive issues in high doses',
            'Consult doctor if taking thyroid medication',
            'Avoid during pregnancy without consultation',
            'May interact with sedative medications',
            'Start with small doses',
            'May cause nausea in some',
            'Not for excessive use',
            'May slow heart rate in high doses'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Tikta, Madhura (Bitter, Sweet)',
            'Guna (Quality)': 'Laghu, Snigdha (Light, Unctuous)',
            'Virya (Potency)': 'Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Madhura (Sweet)',
            'Dosha Effect': 'Balances all three Doshas',
            'Prabhava (Special)': 'Medhya (Brain tonic), Rasayana (Rejuvenative)'
        }
    },
    'turmeric': {
        'common_name': 'Turmeric',
        'scientific_name': 'Curcuma longa',
        'hindi_name': 'हल्दी',
        'family': 'Zingiberaceae',
        'description': 'Turmeric is a flowering plant of the ginger family. It is widely used as a spice and has powerful medicinal properties. It contains curcumin, a potent anti-inflammatory compound. Known as the "Golden Spice of Life".',
        'medicinal_uses': [
            'Powerful anti-inflammatory for joints',
            'Wound healing and antiseptic',
            'Digestive aid and metabolism booster',
            'Skin disorders and complexion',
            'Respiratory issues and asthma',
            'Liver detoxification',
            'Anti-cancer properties (research)',
            'Arthritis and pain relief',
            'Alzheimer\'s prevention',
            'Depression management'
        ],
        'health_benefits': [
            'Powerful antioxidant and anti-inflammatory',
            'Brain health and cognitive function',
            'Heart health and cholesterol management',
            'Arthritis relief and joint health',
            'Cancer prevention (research)',
            'Immune system support',
            'Digestive health',
            'Liver protective',
            'Anti-aging effects',
            'Mood enhancement'
        ],
        'how_to_use': [
            'Golden milk: turmeric with warm milk',
            'In cooking - curries and rice',
            'Turmeric paste for wounds and skin',
            'Turmeric with honey for cough',
            'Turmeric supplements with piperine',
            'Turmeric tea for health',
            'In face packs for glowing skin',
            'Turmeric gargle for sore throat',
            'Turmeric water for detox',
            'Turmeric oil for massage'
        ],
        'precautions': [
            'Avoid with blood thinners (may increase bleeding risk)',
            'May cause stomach upset in high doses',
            'Avoid in gallstones and bile duct obstruction',
            'Not for iron deficiency without supervision',
            'May lower blood pressure',
            'Consult before surgery',
            'May cause allergies in some',
            'May interfere with chemotherapy'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Tikta, Katu (Bitter, Pungent)',
            'Guna (Quality)': 'Laghu, Ruksha (Light, Dry)',
            'Virya (Potency)': 'Ushna (Heating)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Kapha and Vata',
            'Prabhava (Special)': 'Kushtaghna (Anti-skin disorder), Varnya (Improves complexion)'
        }
    },
    'ginger': {
        'common_name': 'Ginger',
        'scientific_name': 'Zingiber officinale',
        'hindi_name': 'अदरक',
        'family': 'Zingiberaceae',
        'description': 'Ginger is a flowering plant widely used as a spice and for its medicinal properties. It is a common ingredient in Ayurvedic medicine for digestive and respiratory health. Known as the "Universal Medicine" in Ayurveda.',
        'medicinal_uses': [
            'Nausea relief - motion sickness, morning sickness',
            'Digestive aid - indigestion, gas, bloating',
            'Cold and flu - reduces symptoms',
            'Menstrual pain relief',
            'Inflammation reduction - arthritis, muscle pain',
            'Respiratory health - cough, asthma',
            'Circulation improvement',
            'Antimicrobial properties',
            'Migraine headache relief',
            'Lowering cholesterol'
        ],
        'health_benefits': [
            'Settles stomach and reduces nausea',
            'Reduces muscle pain and soreness',
            'Lowers blood sugar levels',
            'Lowers cholesterol naturally',
            'Antibacterial and antiviral',
            'Anti-inflammatory effects',
            'Digestive enzyme stimulant',
            'Warming effect on body',
            'Antioxidant properties',
            'Improves brain function'
        ],
        'how_to_use': [
            'Ginger tea: fresh ginger boiled in water',
            'Fresh ginger in cooking and curries',
            'Ginger juice with honey for cough',
            'Dried ginger powder (sonth) with warm water',
            'Ginger candy for nausea',
            'Ginger compress for pain',
            'In soups and broths',
            'Ginger pickle with meals',
            'Ginger oil for massage',
            'Ginger steam for congestion'
        ],
        'precautions': [
            'May cause heartburn in sensitive people',
            'Avoid with bleeding disorders',
            'May lower blood pressure',
            'Consult before surgery (may increase bleeding)',
            'Avoid in gallstones without supervision',
            'Start with small doses',
            'Not for excessive consumption',
            'May interact with blood thinners'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Katu (Pungent)',
            'Guna (Quality)': 'Laghu, Snigdha (Light, Unctuous)',
            'Virya (Potency)': 'Ushna (Heating)',
            'Vipaka (Post-digestive)': 'Madhura (Sweet)',
            'Dosha Effect': 'Balances Kapha and Vata',
            'Prabhava (Special)': 'Deepana (Digestive stimulant), Rochana (Appetizer)'
        }
    },
    'curry_leaf': {
        'common_name': 'Curry Leaf',
        'scientific_name': 'Murraya koenigii',
        'hindi_name': 'कढ़ी पत्ता',
        'family': 'Rutaceae',
        'description': 'Curry leaves are aromatic herbs used in cooking with significant medicinal properties. They are rich in iron and antioxidants, making them valuable for health. An essential part of South Indian cuisine and medicine.',
        'medicinal_uses': [
            'Aids digestion and relieves nausea',
            'Promotes hair growth and prevents graying',
            'Lowers blood sugar levels',
            'Improves eyesight and vision',
            'Reduces cholesterol naturally',
            'Anemia prevention (high in iron)',
            'Anti-inflammatory effects',
            'Liver protective',
            'Weight management',
            'Morning sickness relief'
        ],
        'health_benefits': [
            'Rich in iron and folic acid',
            'Contains antioxidants that fight free radicals',
            'Anti-diabetic properties',
            'Supports weight loss',
            'Liver protective effects',
            'Improves hair health',
            'Digestive stimulant',
            'Antimicrobial properties',
            'Rich in Vitamin A and calcium',
            'Anti-aging effects'
        ],
        'how_to_use': [
            'Add fresh leaves to curries and dishes',
            'Make curry leaf chutney for digestion',
            'Use curry leaf oil for hair massage',
            'Drink curry leaf tea for diabetes',
            'Curry leaf powder with buttermilk',
            'Chew fresh leaves daily',
            'In soups and stews',
            'Curry leaf paste for hair',
            'Curry leaf juice for weight loss',
            'Dried curry leaves in powders'
        ],
        'precautions': [
            'Generally safe when used in cooking',
            'Medicinal doses should be monitored',
            'May lower blood sugar significantly',
            'Consult doctor if taking diabetes medication',
            'May cause allergies in some',
            'Moderate consumption recommended',
            'Wash thoroughly before use',
            'Avoid in excessive amounts'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Katu, Tikta (Pungent, Bitter)',
            'Guna (Quality)': 'Laghu, Ruksha (Light, Dry)',
            'Virya (Potency)': 'Ushna (Heating)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Kapha and Vata',
            'Prabhava (Special)': 'Keshya (Hair tonic), Deepana (Digestive)'
        }
    },
    'hibiscus': {
        'common_name': 'Hibiscus (Gudhal)',
        'scientific_name': 'Hibiscus rosa-sinensis',
        'hindi_name': 'गुड़हल',
        'family': 'Malvaceae',
        'description': 'Hibiscus is known for its beautiful flowers and significant medicinal properties for hair and heart health. It is rich in antioxidants and Vitamin C. The flowers are used in worship and traditional medicine.',
        'medicinal_uses': [
            'Promotes hair growth and prevents graying',
            'Lowers blood pressure naturally',
            'Supports liver health',
            'Relieves menstrual cramps',
            'Improves skin health and complexion',
            'Anti-inflammatory effects',
            'Diuretic properties',
            'Cholesterol management',
            'Cough and cold relief',
            'Weight loss aid'
        ],
        'health_benefits': [
            'Rich in antioxidants and Vitamin C',
            'Natural diuretic properties',
            'Lowers cholesterol levels',
            'Anti-inflammatory effects',
            'Hair conditioning properties',
            'Cooling effect on body',
            'Supports cardiovascular health',
            'Anti-aging effects',
            'Improves metabolism',
            'Antimicrobial properties'
        ],
        'how_to_use': [
            'Use hibiscus powder in hair packs',
            'Drink hibiscus tea for blood pressure',
            'Apply hibiscus paste on hair for growth',
            'Use flower extract in skin care',
            'Hibiscus juice for hair',
            'Hibiscus oil for scalp massage',
            'In herbal hair oils',
            'Hibiscus face pack for glow',
            'Hibiscus shampoo',
            'Hibiscus conditioner'
        ],
        'precautions': [
            'Avoid during pregnancy',
            'May lower blood pressure significantly',
            'Consult doctor if taking blood pressure medication',
            'May interact with diabetes medications',
            'Start with small doses',
            'May cause allergies in some',
            'Monitor blood pressure regularly',
            'Avoid in hypotension'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Amla, Kashaya (Sour, Astringent)',
            'Guna (Quality)': 'Laghu, Ruksha (Light, Dry)',
            'Virya (Potency)': 'Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Amla (Sour)',
            'Dosha Effect': 'Balances Pitta and Kapha',
            'Prabhava (Special)': 'Keshya (Hair tonic), Hridya (Heart tonic)'
        }
    },
    'mint': {
        'common_name': 'Mint (Pudina)',
        'scientific_name': 'Mentha spicata',
        'hindi_name': 'पुदीना',
        'family': 'Lamiaceae',
        'description': 'Mint is a refreshing herb with cooling properties and numerous health benefits. It is widely used for digestive issues and oral health. Known for its characteristic aroma and taste.',
        'medicinal_uses': [
            'Relieves indigestion and IBS symptoms',
            'Clears respiratory congestion',
            'Soothes headaches and migraines',
            'Freshens breath naturally',
            'Relieves muscle pain and spasms',
            'Anti-nausea and anti-emetic',
            'Cooling effect on body',
            'Skin conditions and itching',
            'Stress and anxiety relief',
            'Nasal congestion'
        ],
        'health_benefits': [
            'Antioxidant and anti-inflammatory properties',
            'Antibacterial effects for oral health',
            'Calms stomach muscles and relieves gas',
            'Cooling effect on body',
            'Analgesic properties for pain',
            'Improves digestion',
            'Respiratory health',
            'Mental alertness',
            'Rich in menthol',
            'Appetite stimulant'
        ],
        'how_to_use': [
            'Chew fresh leaves for fresh breath',
            'Make mint tea for digestion',
            'Apply mint paste for headache relief',
            'Use in salads and chutneys',
            'Mint juice with lemon',
            'In smoothies and drinks',
            'Mint oil for aromatherapy',
            'Pudina chutney with meals',
            'Mint water for summer',
            'Steam inhalation for cold'
        ],
        'precautions': [
            'Generally safe in food amounts',
            'Large amounts may cause heartburn',
            'Avoid in infants and young children',
            'May interact with certain medications',
            'May cause allergies in some',
            'Avoid with GERD in high doses',
            'Consult for medicinal use',
            'May irritate mouth ulcers'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Katu (Pungent)',
            'Guna (Quality)': 'Laghu, Ruksha (Light, Dry)',
            'Virya (Potency)': 'Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Pitta and Kapha',
            'Prabhava (Special)': 'Deepana (Digestive), Rochana (Appetizer)'
        }
    },
    'lemon': {
        'common_name': 'Lemon (Nimbu)',
        'scientific_name': 'Citrus limon',
        'hindi_name': 'नींबू',
        'family': 'Rutaceae',
        'description': 'Lemon is a citrus fruit rich in Vitamin C with powerful detoxifying properties. It is widely used in Ayurveda for its cleansing and digestive benefits. A staple in every Indian household.',
        'medicinal_uses': [
            'Boosts immunity and prevents scurvy',
            'Aids digestion and weight loss',
            'Purifies blood and detoxifies body',
            'Improves skin health and complexion',
            'Prevents kidney stones',
            'Respiratory health - cough, cold',
            'Antimicrobial properties',
            'Alkalizes body after digestion',
            'Sore throat relief',
            'Reduces inflammation'
        ],
        'health_benefits': [
            'High in Vitamin C and antioxidants',
            'Alkalizing effect on body',
            'Supports liver detoxification',
            'Antibacterial and antiviral properties',
            'Rich in potassium and minerals',
            'Improves iron absorption',
            'Hydrates and energizes',
            'Skin brightening effects',
            'Weight management',
            'Heart health'
        ],
        'how_to_use': [
            'Drink warm lemon water every morning',
            'Use lemon juice in salads and dishes',
            'Apply lemon juice on skin for glow',
            'Use lemon with honey for sore throat',
            'Lemon tea for health',
            'Lemon juice with warm water for detox',
            'In pickles and preserves',
            'Lemon zest for flavor',
            'Lemonade for hydration',
            'Lemon juice for hair rinse'
        ],
        'precautions': [
            'May erode tooth enamel - rinse mouth after use',
            'Can cause heartburn in some people',
            'Dilute properly before consumption',
            'Avoid on open wounds',
            'May interact with certain medications',
            'Use in moderation',
            'Avoid excessive consumption',
            'May increase sun sensitivity'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Amla (Sour)',
            'Guna (Quality)': 'Laghu, Snigdha (Light, Unctuous)',
            'Virya (Potency)': 'Ushna (Heating)',
            'Vipaka (Post-digestive)': 'Amla (Sour)',
            'Dosha Effect': 'Balances Kapha and Vata, increases Pitta in excess',
            'Prabhava (Special)': 'Deepana (Digestive), Pachana (Digestive)'
        }
    },
    'pomegranate': {
        'common_name': 'Pomegranate (Anar)',
        'scientific_name': 'Punica granatum',
        'hindi_name': 'अनार',
        'family': 'Lythraceae',
        'description': 'Pomegranate is a superfruit packed with antioxidants and numerous health benefits for heart and overall health. It is considered a sacred fruit in many cultures and is mentioned in ancient texts.',
        'medicinal_uses': [
            'Improves heart health and circulation',
            'Lowers blood pressure naturally',
            'Fights cancer cells (research)',
            'Improves digestion',
            'Boosts immunity',
            'Anti-inflammatory effects',
            'Diabetes management',
            'Oral health and gum disease',
            'Diarrhea and dysentery',
            'Anemia prevention'
        ],
        'health_benefits': [
            'Extremely high in antioxidants (punicalagins)',
            'Anti-inflammatory properties',
            'Rich in Vitamin C and K',
            'Supports joint health',
            'Improves memory and brain function',
            'Heart protective',
            'Lowers cholesterol',
            'Anti-aging effects',
            'Improves exercise performance',
            'Antimicrobial properties'
        ],
        'how_to_use': [
            'Eat fresh pomegranate seeds daily',
            'Drink pomegranate juice for heart health',
            'Use in salads and desserts',
            'Apply pomegranate paste on skin',
            'Pomegranate molasses in cooking',
            'Anar juice with meals',
            'Dried seeds as snack',
            'In smoothies and bowls',
            'Pomegranate peel tea',
            'Pomegranate face mask'
        ],
        'precautions': [
            'May interact with blood pressure medications',
            'High in natural sugars - moderate if diabetic',
            'Some people may be allergic',
            'May affect certain cholesterol medications',
            'Avoid if you have low blood pressure',
            'Consult before surgery',
            'Moderate consumption recommended',
            'May cause digestive issues in excess'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Madhura, Amla, Kashaya (Sweet, Sour, Astringent)',
            'Guna (Quality)': 'Laghu, Snigdha (Light, Unctuous)',
            'Virya (Potency)': 'Anushna Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Madhura (Sweet)',
            'Dosha Effect': 'Balances Pitta and Kapha',
            'Prabhava (Special)': 'Hridya (Heart tonic), Rasayana (Rejuvenative)'
        }
    },
    'betel': {
        'common_name': 'Betel Leaf (Paan)',
        'scientific_name': 'Piper betle',
        'hindi_name': 'पान',
        'family': 'Piperaceae',
        'description': 'Betel leaf has digestive and medicinal properties, traditionally used in Ayurveda. It is often used as a mouth freshener and digestive aid after meals. An integral part of Indian culture and traditions.',
        'medicinal_uses': [
            'Improves digestion and appetite',
            'Respiratory problems relief',
            'Wound healing and antiseptic',
            'Oral health and fresh breath',
            'Headache and pain relief',
            'Anti-inflammatory effects',
            'Aphrodisiac properties',
            'Constipation relief',
            'Joint pain management',
            'Skin disorders'
        ],
        'health_benefits': [
            'Antimicrobial properties',
            'Antioxidant effects',
            'Anti-inflammatory benefits',
            'Digestive stimulant',
            'Respiratory health support',
            'Oral hygiene',
            'Pain relief',
            'Cooling effect',
            'Rich in vitamins',
            'Immunomodulatory effects'
        ],
        'how_to_use': [
            'Chew fresh leaf after meals for digestion',
            'Apply leaf paste on wounds',
            'Use in aromatherapy for headaches',
            'Betel leaf juice for cough',
            'Paan with digestive spices',
            'Warm leaf application for pain',
            'Betel oil for oral health',
            'Leaf decoction for respiratory issues',
            'Betel leaf water for bathing',
            'In traditional ceremonies'
        ],
        'precautions': [
            'Avoid with tobacco and areca nut',
            'May cause allergies in some',
            'Moderate use recommended',
            'Consult doctor for medicinal use',
            'Avoid during pregnancy',
            'May cause mouth ulcers in some',
            'Not for long-term excessive use',
            'May stain teeth'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Katu, Tikta (Pungent, Bitter)',
            'Guna (Quality)': 'Laghu, Ruksha (Light, Dry)',
            'Virya (Potency)': 'Ushna (Heating)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Kapha and Vata',
            'Prabhava (Special)': 'Varnya (Improves complexion), Kanthya (Throat tonic)'
        }
    },
    'papaya': {
        'common_name': 'Papaya',
        'scientific_name': 'Carica papaya',
        'hindi_name': 'पपीता',
        'family': 'Caricaceae',
        'description': 'Papaya is a tropical fruit rich in digestive enzymes and antioxidants, known for its numerous health benefits. It contains papain, a powerful digestive enzyme. Called the "Fruit of the Angels" by Christopher Columbus.',
        'medicinal_uses': [
            'Improves digestion and relieves constipation',
            'Treats skin wounds and burns',
            'Boosts immunity with Vitamin C',
            'Supports heart health',
            'Anti-parasitic properties',
            'Menstrual pain relief',
            'Anti-inflammatory effects',
            'Dengue fever treatment (leaf juice)',
            'Anti-cancer properties (research)',
            'Liver protective'
        ],
        'health_benefits': [
            'Contains papain enzyme for protein digestion',
            'Rich in Vitamin C and antioxidants',
            'High fiber content for digestive health',
            'Anti-inflammatory properties',
            'Wound healing capabilities',
            'Immune booster',
            'Heart protective',
            'Skin health improvement',
            'Rich in Vitamin A for eyes',
            'Anti-aging effects'
        ],
        'how_to_use': [
            'Eat ripe papaya for digestion',
            'Apply raw papaya on wounds and burns',
            'Papaya seed juice for parasites',
            'Papaya leaf tea for dengue fever',
            'Papaya smoothie for breakfast',
            'Green papaya in salads',
            'Papaya face mask for skin',
            'Papaya with honey for digestive health',
            'Papaya enzyme supplements',
            'Papaya for tenderizing meat'
        ],
        'precautions': [
            'Unripe papaya may cause uterine contractions - avoid during pregnancy',
            'Papaya seeds in large amounts may be toxic',
            'May cause allergies in latex-sensitive individuals',
            'Moderate consumption recommended',
            'Avoid if allergic to latex',
            'Consult doctor for medicinal use',
            'Start with small amounts',
            'May interfere with blood thinners'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Madhura (Sweet)',
            'Guna (Quality)': 'Guru, Snigdha (Heavy, Unctuous)',
            'Virya (Potency)': 'Ushna (Heating)',
            'Vipaka (Post-digestive)': 'Madhura (Sweet)',
            'Dosha Effect': 'Balances Vata and Kapha',
            'Prabhava (Special)': 'Deepana (Digestive), Pachana (Digestive)'
        }
    },
    'mango': {
        'common_name': 'Mango (Aam)',
        'scientific_name': 'Mangifera indica',
        'hindi_name': 'आम',
        'family': 'Anacardiaceae',
        'description': 'Mango is the king of fruits with numerous health benefits beyond its delicious taste. It is rich in vitamins, minerals, and antioxidants. Known as the "National Fruit of India" and loved worldwide.',
        'medicinal_uses': [
            'Boosts immunity with high Vitamin C',
            'Promotes eye health with Vitamin A',
            'Aids digestion and prevents constipation',
            'Lowers cholesterol',
            'Alkalizes whole body',
            'Improves skin health',
            'Energy booster',
            'Cooling effect in summer',
            'Gut health improvement',
            'Memory enhancement'
        ],
        'health_benefits': [
            'Rich in vitamins A, C, and E',
            'High in fiber for digestive health',
            'Contains antioxidants like quercetin',
            'Supports heart health',
            'Anti-cancer properties',
            'Improves brain function',
            'Enhances iron absorption',
            'Skin and hair health',
            'Boosts immune system',
            'Alkaline-forming food'
        ],
        'how_to_use': [
            'Eat ripe mangoes in season',
            'Use raw mango in chutneys and pickles',
            'Drink mango juice for energy',
            'Apply mango pulp on skin for glow',
            'Mango lassi for digestive health',
            'Aam panna for heat stroke',
            'Mango smoothie bowls',
            'Dried mango as snack',
            'Mango salsa',
            'Mango desserts'
        ],
        'precautions': [
            'High in natural sugars - moderate if diabetic',
            'Some people may be allergic to mango skin',
            'Avoid unripe mangoes in large quantities',
            'Wash thoroughly before eating',
            'May cause acidity in some',
            'Moderate consumption recommended',
            'Avoid with certain medications',
            'May cause heat in body'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Madhura, Amla (Sweet, Sour - unripe)',
            'Guna (Quality)': 'Guru, Snigdha (Heavy, Unctuous)',
            'Virya (Potency)': 'Ushna (Heating)',
            'Vipaka (Post-digestive)': 'Madhura (Sweet)',
            'Dosha Effect': 'Balances Vata, increases Pitta in excess',
            'Prabhava (Special)': 'Balya (Strength promoter), Vrishya (Aphrodisiac)'
        }
    },
    'guava': {
        'common_name': 'Guava',
        'scientific_name': 'Psidium guajava',
        'hindi_name': 'अमरूद',
        'family': 'Myrtaceae',
        'description': 'Guava is a tropical fruit rich in vitamins and antioxidants with numerous health benefits. It has more Vitamin C than oranges and is packed with fiber. A humble fruit with extraordinary health benefits.',
        'medicinal_uses': [
            'Treats diarrhea and dysentery',
            'Manages blood sugar levels',
            'Improves heart health',
            'Boosts immunity',
            'Aids weight loss',
            'Constipation relief (with seeds)',
            'Respiratory health',
            'Skin health improvement',
            'Eye health',
            'Thyroid function support'
        ],
        'health_benefits': [
            'Rich in Vitamin C (4x more than orange)',
            'High fiber content for digestion',
            'Low glycemic index for diabetics',
            'Potassium for blood pressure',
            'Anti-inflammatory properties',
            'Immune booster',
            'Eye health (Vitamin A)',
            'Brain function support',
            'Antioxidant properties',
            'Lycopene for heart health'
        ],
        'how_to_use': [
            'Eat fresh fruit for vitamins',
            'Guava leaf tea for diarrhea',
            'Leaf extract for diabetes',
            'Fruit in salads and juices',
            'Guava smoothie',
            'Guava jelly and preserves',
            'Raw guava with salt and pepper',
            'Guava juice for immunity',
            'Guava leaf paste for wounds',
            'Guava for weight loss'
        ],
        'precautions': [
            'May cause bloating if eaten in excess',
            'Monitor blood sugar if diabetic',
            'Eat ripe fruit for best benefits',
            'Wash thoroughly before eating',
            'Seeds may cause constipation in some',
            'Moderate consumption recommended',
            'Consult for medicinal leaf use',
            'May interact with diabetes meds'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Madhura, Kashaya (Sweet, Astringent)',
            'Guna (Quality)': 'Guru, Ruksha (Heavy, Dry)',
            'Virya (Potency)': 'Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Madhura (Sweet)',
            'Dosha Effect': 'Balances Pitta and Kapha',
            'Prabhava (Special)': 'Grahi (Absorbent), Stambhana (Anti-diarrheal)'
        }
    },
    'lemon_grass': {
        'common_name': 'Lemon Grass',
        'scientific_name': 'Cymbopogon citratus',
        'hindi_name': 'लेमन ग्रास',
        'family': 'Poaceae',
        'description': 'Lemon grass is an aromatic herb with refreshing citrus flavor and medicinal properties. It is widely used in teas and Asian cuisine. Known for its calming and detoxifying effects.',
        'medicinal_uses': [
            'Digestive issues and bloating',
            'Fever and infection reduction',
            'Stress and anxiety relief',
            'Cholesterol management',
            'Detoxification and cleansing',
            'Pain relief - headaches, muscle pain',
            'Respiratory health',
            'Insomnia relief',
            'Anti-fungal properties',
            'Weight loss aid'
        ],
        'health_benefits': [
            'Antimicrobial and antibacterial',
            'Anti-inflammatory properties',
            'Rich in antioxidants',
            'Diuretic effects',
            'Analgesic properties',
            'Digestive stimulant',
            'Calms nervous system',
            'Fever reducer',
            'Citronella for insects',
            'Skin health'
        ],
        'how_to_use': [
            'Lemon grass tea for digestion',
            'Essential oil for aromatherapy',
            'Fresh stalks in cooking',
            'Poultice for pain relief',
            'Lemon grass soup for cold',
            'Infused water for detox',
            'In curries and stir-fries',
            'Lemon grass oil for massage',
            'Lemon grass bath for relaxation',
            'Insect repellent spray'
        ],
        'precautions': [
            'Generally safe in food amounts',
            'May cause allergies in some',
            'Avoid during pregnancy in large amounts',
            'Consult doctor for medicinal use',
            'May lower blood pressure',
            'Avoid with kidney disease',
            'Start with small doses',
            'May affect liver enzymes'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Katu, Tikta (Pungent, Bitter)',
            'Guna (Quality)': 'Laghu, Ruksha (Light, Dry)',
            'Virya (Potency)': 'Ushna (Heating)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Kapha and Vata',
            'Prabhava (Special)': 'Jwaraghna (Anti-pyretic), Deepana (Digestive)'
        }
    },
    'jasmine': {
        'common_name': 'Jasmine',
        'scientific_name': 'Jasminum officinale',
        'hindi_name': 'चमेली',
        'family': 'Oleaceae',
        'description': 'Jasmine is renowned for its fragrant flowers and therapeutic properties in aromatherapy. It is used for stress relief and skin care. The "Queen of the Night" for its intoxicating fragrance.',
        'medicinal_uses': [
            'Stress and anxiety relief',
            'Skin care and complexion',
            'Headache and pain relief',
            'Antiseptic for wounds',
            'Mood enhancement',
            'Aphrodisiac properties',
            'Sleep aid',
            'Respiratory health',
            'Depression management',
            'Hormonal balance'
        ],
        'health_benefits': [
            'Antidepressant properties',
            'Antiseptic and antimicrobial',
            'Anti-inflammatory effects',
            'Relaxing and calming',
            'Aphrodisiac properties',
            'Skin soothing',
            'Hormonal balance',
            'Cooling effect',
            'Antispasmodic',
            'Galactagogue (increases milk)'
        ],
        'how_to_use': [
            'Jasmine tea for relaxation',
            'Essential oil for aromatherapy',
            'Flower paste for skin care',
            'Jasmine water as toner',
            'Jasmine oil for massage',
            'Dried flowers in potpourri',
            'Jasmine garland for fragrance',
            'In face packs and creams',
            'Jasmine bath for relaxation',
            'Jasmine perfume'
        ],
        'precautions': [
            'Generally safe in moderation',
            'May cause allergies in some',
            'Essential oil should be diluted',
            'Avoid during pregnancy in large amounts',
            'Patch test before skin application',
            'Consult for medicinal use',
            'Use authentic products only',
            'May cause photosensitivity'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Tikta, Kashaya (Bitter, Astringent)',
            'Guna (Quality)': 'Laghu, Snigdha (Light, Unctuous)',
            'Virya (Potency)': 'Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Pitta and Kapha',
            'Prabhava (Special)': 'Varnya (Improves complexion), Hridya (Heart tonic)'
        }
    },
    'henna': {
        'common_name': 'Henna (Mehndi)',
        'scientific_name': 'Lawsonia inermis',
        'hindi_name': 'मेहंदी',
        'family': 'Lythraceae',
        'description': 'Henna is famous for its natural dyeing properties and cooling medicinal effects. It is used for hair conditioning and skin applications. An essential part of Indian weddings and festivals.',
        'medicinal_uses': [
            'Natural hair dye and conditioner',
            'Treats skin diseases and infections',
            'Cooling effect for headaches and burns',
            'Anti-fungal properties for feet',
            'Soothes inflammatory conditions',
            'Wound healing',
            'Bruises and sprains',
            'Hand and foot care',
            'Dandruff treatment',
            'Body art'
        ],
        'health_benefits': [
            'Natural cooling agent for body',
            'Antibacterial and antifungal properties',
            'Conditions hair and prevents dandruff',
            'Heals wounds and burns',
            'Anti-inflammatory effects',
            'Astringent properties',
            'Hair strengthening',
            'Skin soothing',
            'Nail health',
            'Anti-hemorrhagic'
        ],
        'how_to_use': [
            'Apply henna paste on hair for coloring',
            'Use henna paste on burns for relief',
            'Apply on feet for fungal infections',
            'Use as natural hand and body art',
            'Henna oil for hair growth',
            'Poultice for headaches',
            'Henna pack for skin conditions',
            'Mixed with other herbs for hair',
            'Henna for mehndi designs',
            'Henna conditioner'
        ],
        'precautions': [
            'Test for allergy before use',
            'Avoid chemical mixed henna',
            'May dry hair if used frequently',
            'Use natural henna without additives',
            'Avoid during pregnancy',
            'Keep away from eyes',
            'May cause contact dermatitis in some',
            'Not for consumption'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Kashaya, Tikta (Astringent, Bitter)',
            'Guna (Quality)': 'Laghu, Ruksha (Light, Dry)',
            'Virya (Potency)': 'Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Pitta and Kapha',
            'Prabhava (Special)': 'Keshya (Hair tonic), Varnya (Improves complexion)'
        }
    },
    'sandalwood': {
        'common_name': 'Sandalwood',
        'scientific_name': 'Santalum album',
        'hindi_name': 'चंदन',
        'family': 'Santalaceae',
        'description': 'Sandalwood is prized for its aromatic wood and oil, used in skincare, religious ceremonies, and traditional medicine for its cooling properties. One of the most precious woods in the world.',
        'medicinal_uses': [
            'Skin diseases and inflammation',
            'Fever and headache relief',
            'Urinary tract infections',
            'Cooling effect on body',
            'Acne and pimples treatment',
            'Stress and anxiety relief',
            'Respiratory health',
            'Anti-aging for skin',
            'Burns and sunburn',
            'Meditation aid'
        ],
        'health_benefits': [
            'Anti-inflammatory properties',
            'Antiseptic and antimicrobial',
            'Cooling and soothing',
            'Astringent effects',
            'Antioxidant properties',
            'Calms nervous system',
            'Skin rejuvenation',
            'Meditation aid',
            'Expectorant properties',
            'Diuretic effects'
        ],
        'how_to_use': [
            'Apply sandalwood paste on skin',
            'Sandalwood oil for aromatherapy',
            'In face packs for skin glow',
            'Sandalwood powder with rose water',
            'Chandan tilak for cooling',
            'In incense and perfumes',
            'Sandalwood soap for bathing',
            'Mixed with other herbs',
            'Sandalwood for puja',
            'Sandalwood cream'
        ],
        'precautions': [
            'Generally safe for external use',
            'May cause allergies in some',
            'Use authentic sandalwood only',
            'Avoid during pregnancy for internal use',
            'Patch test before application',
            'Consult for medicinal use',
            'Expensive - beware of adulteration',
            'Not for consumption in large amounts'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Tikta, Madhura (Bitter, Sweet)',
            'Guna (Quality)': 'Laghu, Snigdha (Light, Unctuous)',
            'Virya (Potency)': 'Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Pitta and Kapha',
            'Prabhava (Special)': 'Varnya (Improves complexion), Hridya (Heart tonic)'
        }
    },
    'castor': {
        'common_name': 'Castor Plant',
        'scientific_name': 'Ricinus communis',
        'hindi_name': 'अरंडी',
        'family': 'Euphorbiaceae',
        'description': 'Castor plant is known for its medicinal oil but all parts of plant contain toxic compounds. The oil is widely used in Ayurveda. Handle with extreme caution and respect.',
        'medicinal_uses': [
            'Castor oil for constipation relief',
            'Anti-inflammatory for arthritis',
            'Skin conditions treatment',
            'Hair growth promotion',
            'Labor induction (traditional)',
            'Wound healing',
            'Joint pain relief',
            'Detoxification',
            'Lymphatic stimulation',
            'Eye health (Ayurvedic)'
        ],
        'health_benefits': [
            'Powerful laxative properties',
            'Anti-inflammatory effects',
            'Antimicrobial properties',
            'Moisturizing for skin and hair',
            'Pain relief for joints',
            'Immune modulation',
            'Lymphatic stimulation',
            'Anti-fungal effects',
            'Anti-bacterial',
            'Anti-oxidant'
        ],
        'how_to_use': [
            'Castor oil for constipation (1-2 tsp only)',
            'External application for arthritis',
            'Hair oil for growth and strength',
            'Skin moisturizer and healer',
            'Castor oil packs for liver',
            'Warm oil massage for pain',
            'In herbal formulations',
            'Eye drops (Ayurvedic - under supervision)',
            'Castor oil for eyelashes',
            'Castor oil for wound healing'
        ],
        'precautions': [
            '⚠️ SEEDS ARE HIGHLY TOXIC - never consume',
            'Castor oil only in recommended doses',
            'Avoid during pregnancy',
            'May cause allergic reactions',
            'Consult doctor before internal use',
            'Not for long-term use',
            'Keep away from children',
            'May cause severe diarrhea'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Madhura, Katu, Tikta (Sweet, Pungent, Bitter)',
            'Guna (Quality)': 'Guru, Snigdha, Sukshma (Heavy, Unctuous, Penetrating)',
            'Virya (Potency)': 'Ushna (Heating)',
            'Vipaka (Post-digestive)': 'Madhura (Sweet)',
            'Dosha Effect': 'Balances Vata',
            'Prabhava (Special)': 'Vatanulomana (Vata regulating), Rechana (Purgative)'
        }
    },
    'insulin': {
        'common_name': 'Insulin Plant',
        'scientific_name': 'Costus igneus',
        'hindi_name': 'इन्सुलिन प्लांट',
        'family': 'Costaceae',
        'description': 'Insulin plant is known for its anti-diabetic properties and blood sugar regulating effects. It is named for its ability to help manage diabetes. A natural way to support pancreatic health.',
        'medicinal_uses': [
            'Diabetes management',
            'Blood sugar regulation',
            'Antioxidant properties',
            'Urinary tract health',
            'Liver protection',
            'Weight management',
            'Pancreatic health',
            'Metabolic disorders',
            'Kidney health',
            'Digestive health'
        ],
        'health_benefits': [
            'Lowers blood glucose levels',
            'Rich in antioxidants',
            'Diuretic properties',
            'Anti-inflammatory effects',
            'Hepatoprotective qualities',
            'Improves insulin sensitivity',
            'Pancreatic function support',
            'Metabolism booster',
            'Anti-diabetic properties',
            'Beta-cell regeneration'
        ],
        'how_to_use': [
            'Chew fresh leaves daily',
            'Make leaf tea for diabetes',
            'Leaf powder with water',
            'Consult doctor for dosage',
            'Leaf juice for blood sugar',
            'In herbal formulations',
            'With other anti-diabetic herbs',
            'Monitor blood sugar regularly',
            'Insulin plant capsules',
            'Decoction for diabetes'
        ],
        'precautions': [
            'Monitor blood sugar regularly',
            'Consult doctor before use',
            'May interact with diabetes medication',
            'Start with small doses',
            'Avoid during pregnancy',
            'May cause hypoglycemia',
            'Not a replacement for medication',
            'May cause digestive issues'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Tikta (Bitter)',
            'Guna (Quality)': 'Laghu, Ruksha (Light, Dry)',
            'Virya (Potency)': 'Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Kapha and Pitta',
            'Prabhava (Special)': 'Pramehaghna (Anti-diabetic), Kaphahara (Kapha reducing)'
        }
    },
    'periwinkle': {
        'common_name': 'Periwinkle (Nithyapushpa)',
        'scientific_name': 'Catharanthus roseus',
        'hindi_name': 'सदाबहार',
        'family': 'Apocynaceae',
        'description': 'Periwinkle is known for its beautiful flowers and important medicinal compounds used in cancer treatment. It contains alkaloids with anti-cancer properties. A beautiful flower with powerful medicine.',
        'medicinal_uses': [
            'Diabetes management',
            'Traditional use for cancer',
            'Blood pressure regulation',
            'Antimicrobial properties',
            'Memory enhancement',
            'Wound healing',
            'Menstrual disorders',
            'Sore throat relief',
            'Hodgkin\'s lymphoma (contains vincristine)',
            'Leukemia treatment (contains vinblastine)'
        ],
        'health_benefits': [
            'Anti-diabetic properties',
            'Source of anti-cancer compounds (vincristine, vinblastine)',
            'Antihypertensive effects',
            'Antioxidant properties',
            'Cognitive enhancement',
            'Anti-inflammatory effects',
            'Antimicrobial activity',
            'Traditional use for various ailments',
            'Cytotoxic effects on cancer cells',
            'Immune modulation'
        ],
        'how_to_use': [
            '⚠️ STRICTLY UNDER MEDICAL SUPERVISION',
            'Leaf extract for diabetes',
            'Traditional formulations only',
            'Consult doctor for proper use',
            'Never self-medicate for serious conditions',
            'In herbal combinations',
            'Decoction for medicinal use',
            'Ayurvedic preparations only',
            'Standardized extracts only',
            'Pharmaceutical preparations'
        ],
        'precautions': [
            '⚠️ Contains potent alkaloids',
            '⚠️ Use only under medical supervision',
            '⚠️ May interact with medications',
            '⚠️ NOT for self-treatment of cancer',
            'Avoid during pregnancy',
            'May cause side effects',
            'Consult oncologist if needed',
            'Neurotoxic in high doses'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Tikta (Bitter)',
            'Guna (Quality)': 'Laghu, Ruksha (Light, Dry)',
            'Virya (Potency)': 'Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Kapha and Pitta',
            'Prabhava (Special)': 'Pramehaghna (Anti-diabetic), Kushthaghna (Anti-skin disorder)'
        }
    },
    'black_nightshade': {
        'common_name': 'Black Nightshade (Ganike)',
        'scientific_name': 'Solanum nigrum',
        'hindi_name': 'मकोय',
        'family': 'Solanaceae',
        'description': 'Ganike is a medicinal plant used in traditional medicine with both nutritional and therapeutic values. It is used for various ailments. A common weed with uncommon benefits.',
        'medicinal_uses': [
            'Fever and inflammation reduction',
            'Liver disorders treatment',
            'Skin diseases and wounds',
            'Digestive issues management',
            'Respiratory problems relief',
            'Ulcer treatment',
            'Pain relief',
            'Anti-inflammatory effects',
            'Dropsy and edema',
            'Eye disorders'
        ],
        'health_benefits': [
            'Antipyretic properties',
            'Anti-inflammatory effects',
            'Hepatoprotective qualities',
            'Antioxidant properties',
            'Diuretic effects',
            'Digestive stimulant',
            'Wound healing',
            'Antimicrobial activity',
            'Anti-ulcerogenic',
            'Analgesic properties'
        ],
        'how_to_use': [
            'Cooked leaves as vegetable',
            'Leaf juice for fever',
            'Paste application for skin',
            'Decoction for liver health',
            'Berry juice for digestive issues',
            'In herbal formulations',
            'With other herbs for synergy',
            'Traditional preparations',
            'Leaf poultice for wounds',
            'Fruit for edible purposes'
        ],
        'precautions': [
            'Unripe berries may be toxic',
            'Proper identification required',
            'Cook thoroughly before consumption',
            'Consult expert for medicinal use',
            'Avoid during pregnancy',
            'May cause allergies in some',
            'Moderate use recommended',
            'Contains solanine - toxic in large amounts'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Tikta, Katu (Bitter, Pungent)',
            'Guna (Quality)': 'Laghu, Ruksha (Light, Dry)',
            'Virya (Potency)': 'Ushna (Heating)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Kapha and Vata',
            'Prabhava (Special)': 'Yakrituttejaka (Liver stimulant), Jwaraghna (Anti-pyretic)'
        }
    },
    'indian_beech': {
        'common_name': 'Indian Beech (Hongue)',
        'scientific_name': 'Pongamia pinnata',
        'hindi_name': 'करंज',
        'family': 'Fabaceae',
        'description': 'Hongue is a traditional medicinal plant with various therapeutic applications. It is used in Ayurveda for skin and joint conditions. A versatile tree with multiple benefits.',
        'medicinal_uses': [
            'Skin diseases treatment',
            'Rheumatism and joint pain',
            'Digestive disorders',
            'Respiratory problems',
            'Wound healing',
            'Ulcer treatment',
            'Anti-parasitic',
            'Liver disorders',
            'Diarrhea and dysentery',
            'Fever reduction'
        ],
        'health_benefits': [
            'Anti-inflammatory properties',
            'Antimicrobial effects',
            'Analgesic properties',
            'Antioxidant activity',
            'Wound healing properties',
            'Anti-rheumatic',
            'Skin health',
            'Digestive support',
            'Anti-helminthic',
            'Anti-bacterial'
        ],
        'how_to_use': [
            'Oil application for skin conditions',
            'Leaf paste for wounds',
            'Seed powder for digestive issues',
            'Bark decoction for rheumatism',
            'Root paste for ulcers',
            'In herbal oils',
            'Traditional formulations',
            'External applications only',
            'Poultice for joint pain',
            'Oil for hair and skin'
        ],
        'precautions': [
            'Seeds are toxic if consumed raw',
            'Use only under guidance',
            'May cause skin irritation',
            'Avoid internal use without processing',
            'Consult Ayurvedic practitioner',
            'Patch test before use',
            'Keep away from children',
            'Not for self-medication'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Tikta, Katu (Bitter, Pungent)',
            'Guna (Quality)': 'Laghu, Ruksha (Light, Dry)',
            'Virya (Potency)': 'Ushna (Heating)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Kapha and Vata',
            'Prabhava (Special)': 'Kushthaghna (Anti-skin disorder), Shothahara (Anti-inflammatory)'
        }
    },
    'indian_snakeroot': {
        'common_name': 'Indian Snakeroot (Nagadali)',
        'scientific_name': 'Rauvolfia serpentina',
        'hindi_name': 'सर्पगंधा',
        'family': 'Apocynaceae',
        'description': 'Nagadali is a traditional medicinal plant known for its sedative and antihypertensive properties. It is used in Ayurveda for mental disorders. A powerful medicine requiring expert guidance.',
        'medicinal_uses': [
            'High blood pressure management',
            'Mental disorders and insomnia',
            'Snake bite treatment (traditional)',
            'Anxiety and stress relief',
            'Fever and digestive issues',
            'Schizophrenia (traditional)',
            'Epilepsy',
            'Pain relief',
            'Childbirth (traditional)',
            'Psychosis'
        ],
        'health_benefits': [
            'Antihypertensive properties',
            'Sedative and tranquilizing effects',
            'Antipsychotic properties',
            'Antipyretic effects',
            'Traditional use for various ailments',
            'Nervous system calming',
            'Blood pressure regulation',
            'Mental health support',
            'Anti-arrhythmic',
            'Uterine stimulant'
        ],
        'how_to_use': [
            '⚠️ STRICTLY UNDER MEDICAL SUPERVISION',
            'Ayurvedic formulations only',
            'Never self-medicate',
            'Traditional preparations by experts',
            'Root powder in small doses',
            'In classical Ayurvedic medicines',
            'Consult qualified practitioner',
            'Monitor blood pressure regularly',
            'Decoction under guidance',
            'Only standardized extracts'
        ],
        'precautions': [
            '⚠️ POTENT MEDICINE - Requires expert guidance',
            '⚠️ May cause serious side effects',
            '⚠️ Not for self-medication',
            '⚠️ Monitor blood pressure regularly',
            'May cause depression in high doses',
            'Avoid during pregnancy',
            'Do not combine with other sedatives',
            'May cause bradycardia'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Tikta, Kashaya (Bitter, Astringent)',
            'Guna (Quality)': 'Laghu, Snigdha (Light, Unctuous)',
            'Virya (Potency)': 'Ushna (Heating)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Kapha and Vata',
            'Prabhava (Special)': 'Nidrajanana (Sleep inducing), Vishaghna (Anti-venom)'
        }
    },
    'crown_flower': {
        'common_name': 'Crown Flower (Ekka)',
        'scientific_name': 'Calotropis gigantea',
        'hindi_name': 'आक',
        'family': 'Apocynaceae',
        'description': 'Ekka is a medicinal plant with toxic properties, used cautiously in traditional medicine. It has various therapeutic applications. A plant dedicated to Lord Shiva with powerful medicine.',
        'medicinal_uses': [
            'Skin diseases treatment',
            'Digestive disorders in small doses',
            'Traditional use for asthma',
            'Wound healing properties',
            'Anti-inflammatory effects',
            'Pain relief',
            'Fever reduction',
            'Anti-parasitic',
            'Cough and cold',
            'Rheumatism'
        ],
        'health_benefits': [
            'Antimicrobial properties',
            'Anti-inflammatory effects',
            'Analgesic properties',
            'Antioxidant activity',
            'Traditional pain relief',
            'Wound healing',
            'Respiratory health',
            'Skin conditions',
            'Anti-asthmatic',
            'Anti-cancer (research)'
        ],
        'how_to_use': [
            '⚠️ STRICTLY UNDER EXPERT GUIDANCE',
            'External applications for skin',
            'Traditional formulations only',
            'Never consume raw plant parts',
            'Leaf paste for external use',
            'Flower for specific preparations',
            'In Ayurvedic medicines',
            'Purified forms only',
            'Oil for external use',
            'Fumigation for asthma'
        ],
        'precautions': [
            '⚠️ MILKY LATEX IS HIGHLY TOXIC',
            '⚠️ Never consume without processing',
            '⚠️ Keep away from eyes and mouth',
            '⚠️ Use only under qualified supervision',
            'May cause severe irritation',
            'Avoid during pregnancy',
            'Not for self-medication',
            'Cardiotoxic in high doses'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Tikta, Katu (Bitter, Pungent)',
            'Guna (Quality)': 'Laghu, Ruksha, Tikshna (Light, Dry, Sharp)',
            'Virya (Potency)': 'Ushna (Heating)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Kapha and Vata',
            'Prabhava (Special)': 'Kushthaghna (Anti-skin disorder), Shothahara (Anti-inflammatory)'
        }
    },
    'indian_borage': {
        'common_name': 'Indian Borage (Doddapatre)',
        'scientific_name': 'Coleus amboinicus',
        'hindi_name': 'पथर्चुर',
        'family': 'Lamiaceae',
        'description': 'Doddapatre is an aromatic herb with strong medicinal properties for respiratory and digestive health. It is commonly used in home remedies. A must-have in every kitchen garden.',
        'medicinal_uses': [
            'Treats cough and cold effectively',
            'Relieves asthma and bronchitis',
            'Aids digestion and reduces flatulence',
            'Kidney stone treatment',
            'Skin conditions and wounds',
            'Fever reduction',
            'Urinary tract infections',
            'Headache relief',
            'Earache',
            'Rheumatism'
        ],
        'health_benefits': [
            'Expectorant properties for respiratory issues',
            'Antimicrobial and antibacterial',
            'Anti-inflammatory effects',
            'Rich in vitamins and minerals',
            'Diuretic properties',
            'Digestive stimulant',
            'Pain relief',
            'Anti-spasmodic',
            'Anti-tussive',
            'Anti-oxidant'
        ],
        'how_to_use': [
            'Leaf juice with honey for cough',
            'Chew leaves for digestive issues',
            'Apply leaf paste on wounds',
            'Make tea for respiratory problems',
            'In chutneys and salads',
            'Steam inhalation for congestion',
            'Leaf extract for kidney stones',
            'Traditional home remedies',
            'Leaf for earache',
            'Poultice for rheumatism'
        ],
        'precautions': [
            'Avoid during pregnancy',
            'May cause skin irritation in some',
            'Moderate use recommended',
            'Consult doctor for kidney problems',
            'May cause allergies in sensitive individuals',
            'Start with small doses',
            'Not for long-term excessive use',
            'May interact with medications'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Katu, Tikta (Pungent, Bitter)',
            'Guna (Quality)': 'Laghu, Ruksha (Light, Dry)',
            'Virya (Potency)': 'Ushna (Heating)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Kapha and Vata',
            'Prabhava (Special)': 'Kasahara (Anti-cough), Shwasahara (Anti-asthma)'
        }
    },
    'malabar_spinach': {
        'common_name': 'Malabar Spinach (Basale)',
        'scientific_name': 'Basella alba',
        'hindi_name': 'पोई साग',
        'family': 'Basellaceae',
        'description': 'Basale is a nutritious leafy vegetable with cooling properties and medicinal benefits. It is commonly used in Indian cooking. A powerhouse of nutrition with medicinal value.',
        'medicinal_uses': [
            'Treats constipation and digestive issues',
            'Cooling effect on body',
            'Rich in iron for anemia',
            'Promotes wound healing',
            'Anti-inflammatory properties',
            'Urinary health',
            'Skin conditions',
            'Burns and scalds',
            'Diuretic',
            'Laxative'
        ],
        'health_benefits': [
            'High in vitamins A, C, and iron',
            'Rich in fiber for digestion',
            'Mucilaginous properties soothe digestion',
            'Low in calories for weight management',
            'Antioxidant properties',
            'Cooling effect',
            'Blood building',
            'Hydrating',
            'Calcium for bones',
            'Magnesium for nerves'
        ],
        'how_to_use': [
            'Cook as vegetable curry',
            'Make basale juice for constipation',
            'Use in soups and stews',
            'Apply leaf paste on wounds',
            'In salads and stir-fries',
            'Basale dal for nutrition',
            'Leaf juice for anemia',
            'Traditional preparations',
            'Basale raita',
            'Basale with coconut'
        ],
        'precautions': [
            'Generally safe when cooked',
            'May cause oxalate issues in sensitive people',
            'Cook properly to reduce oxalates',
            'Moderate consumption recommended',
            'Wash thoroughly before use',
            'Consult for kidney stone history',
            'Avoid in large raw amounts',
            'May cause bloating in some'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Madhura, Kashaya (Sweet, Astringent)',
            'Guna (Quality)': 'Guru, Snigdha (Heavy, Unctuous)',
            'Virya (Potency)': 'Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Madhura (Sweet)',
            'Dosha Effect': 'Balances Pitta and Vata',
            'Prabhava (Special)': 'Vrushya (Aphrodisiac), Balya (Strength promoter)'
        }
    },
    'bamboo': {
        'common_name': 'Bamboo',
        'scientific_name': 'Bambusoideae',
        'hindi_name': 'बांस',
        'family': 'Poaceae',
        'description': 'Bamboo has various medicinal uses, especially bamboo shoots and leaves in traditional medicine. It is rich in silica and nutrients. The "Green Gold" of the forest with many benefits.',
        'medicinal_uses': [
            'Respiratory disorders treatment',
            'Wound healing and skin conditions',
            'Arthritis and joint pain relief',
            'Digestive health improvement',
            'Fever and infection management',
            'Bone health',
            'Urinary tract infections',
            'Hair health',
            'Anti-parasitic',
            'Anti-inflammatory'
        ],
        'health_benefits': [
            'Rich in silica for bone health',
            'Antioxidant properties',
            'Anti-inflammatory effects',
            'High in dietary fiber',
            'Low calorie nutrient source',
            'Joint support',
            'Skin health',
            'Hair strengthening',
            'Nail health',
            'Detoxification'
        ],
        'how_to_use': [
            'Bamboo shoot curry for digestion',
            'Bamboo leaf tea for respiratory issues',
            'Bamboo sap for skin conditions',
            'Bamboo salt for cooking',
            'Bamboo silica supplements',
            'In soups and stir-fries',
            'Bamboo vinegar for skin',
            'Traditional preparations',
            'Bamboo ash for teeth',
            'Bamboo water for health'
        ],
        'precautions': [
            'Proper cooking required for shoots',
            'Some species may contain toxins',
            'Consult expert for medicinal use',
            'Avoid raw bamboo consumption',
            'May contain cyanide in raw form',
            'Cook thoroughly before eating',
            'Moderate consumption recommended',
            'May cause allergies'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Madhura, Kashaya (Sweet, Astringent)',
            'Guna (Quality)': 'Laghu, Ruksha (Light, Dry)',
            'Virya (Potency)': 'Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Madhura (Sweet)',
            'Dosha Effect': 'Balances Pitta and Kapha',
            'Prabhava (Special)': 'Vrushya (Aphrodisiac), Mutrala (Diuretic)'
        }
    },
    'avocado': {
        'common_name': 'Avocado',
        'scientific_name': 'Persea americana',
        'hindi_name': 'एवोकाडो',
        'family': 'Lauraceae',
        'description': 'Avocado is a nutrient-dense fruit rich in healthy fats, vitamins, and minerals. It is valued for its health benefits and medicinal properties. The "Butter Fruit" of the tropics.',
        'medicinal_uses': [
            'Supports heart health and cholesterol',
            'Promotes skin and hair health',
            'Aids weight management',
            'Improves digestion',
            'Rich source of antioxidants',
            'Anti-inflammatory effects',
            'Eye health',
            'Brain function',
            'Arthritis relief',
            'Pregnancy nutrition'
        ],
        'health_benefits': [
            'High in healthy monounsaturated fats',
            'Rich in fiber for digestive health',
            'Contains vitamins E, C, K, and B6',
            'Potassium-rich for blood pressure',
            'Anti-inflammatory properties',
            'Heart protective',
            'Skin moisturizing',
            'Nutrient absorption',
            'Lutein for eyes',
            'Folate for pregnancy'
        ],
        'how_to_use': [
            'Eat fresh avocado as fruit',
            'Use in salads and sandwiches',
            'Make avocado smoothies',
            'Apply avocado paste on skin and hair',
            'Avocado oil for cooking',
            'Guacamole for healthy snack',
            'In face masks',
            'With honey for hair mask',
            'Avocado toast',
            'Avocado dessert'
        ],
        'precautions': [
            'High in calories - moderate consumption',
            'May cause allergies in some people',
            'Avoid if allergic to latex',
            'Monitor portion size for weight management',
            'May interact with blood thinners',
            'Consult for medicinal use',
            'Moderate consumption recommended',
            'May cause digestive issues in excess'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Madhura (Sweet)',
            'Guna (Quality)': 'Guru, Snigdha (Heavy, Unctuous)',
            'Virya (Potency)': 'Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Madhura (Sweet)',
            'Dosha Effect': 'Balances Vata and Pitta',
            'Prabhava (Special)': 'Balya (Strength promoter), Vrushya (Aphrodisiac)'
        }
    },
    'sapota': {
        'common_name': 'Sapodilla (Chikoo)',
        'scientific_name': 'Manilkara zapota',
        'hindi_name': 'चीकू',
        'family': 'Sapotaceae',
        'description': 'Sapota is a sweet fruit with numerous health benefits, rich in nutrients and dietary fiber. It is enjoyed fresh and in desserts. The "Chocolate Pudding Fruit" of India.',
        'medicinal_uses': [
            'Treats constipation and digestive issues',
            'Boosts energy and prevents anemia',
            'Supports bone health',
            'Anti-inflammatory properties',
            'Improves vision health',
            'Cold and cough relief',
            'Diuretic effects',
            'Skin health',
            'Immune booster',
            'Anti-aging'
        ],
        'health_benefits': [
            'High in dietary fiber for digestion',
            'Rich in iron for anemia prevention',
            'Contains calcium for bone health',
            'Antioxidant properties',
            'Natural energy booster',
            'Vitamin C for immunity',
            'Anti-inflammatory',
            'Cooling effect',
            'Vitamin A for eyes',
            'Potassium for heart'
        ],
        'how_to_use': [
            'Eat ripe fruit as snack',
            'Use in milkshakes and desserts',
            'Apply fruit pulp on skin',
            'Leaf decoction for fever',
            'Chikoo smoothie',
            'In fruit salads',
            'Chikoo ice cream',
            'Traditional preparations',
            'Chikoo halwa',
            'Dried chikoo'
        ],
        'precautions': [
            'High in natural sugars - moderate if diabetic',
            'Unripe fruit may cause mouth irritation',
            'May cause allergies in sensitive individuals',
            'Consume in moderation',
            'Wash thoroughly before eating',
            'Avoid seeds as they are hard',
            'Monitor blood sugar if diabetic',
            'May cause bloating'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Madhura (Sweet)',
            'Guna (Quality)': 'Guru, Snigdha (Heavy, Unctuous)',
            'Virya (Potency)': 'Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Madhura (Sweet)',
            'Dosha Effect': 'Balances Vata and Pitta',
            'Prabhava (Special)': 'Balya (Strength promoter), Vrushya (Aphrodisiac)'
        }
    },
    'noni': {
        'common_name': 'Noni Fruit',
        'scientific_name': 'Morinda citrifolia',
        'hindi_name': 'नोनी',
        'family': 'Rubiaceae',
        'description': 'Noni fruit is known for its immune-boosting properties and has been used in traditional Polynesian medicine for centuries. The "Painkiller Tree" of the Pacific.',
        'medicinal_uses': [
            'Boosts immune system function',
            'Reduces inflammation and pain',
            'Improves skin health and conditions',
            'Supports cardiovascular health',
            'Aids digestion and gut health',
            'Anti-cancer properties (research)',
            'Joint pain relief',
            'Energy booster',
            'Diabetes management',
            'Anti-aging'
        ],
        'health_benefits': [
            'Rich in antioxidants and phytochemicals',
            'Anti-inflammatory properties',
            'Antimicrobial and antibacterial effects',
            'Analgesic (pain-relieving) properties',
            'Immune-modulating effects',
            'Cell regeneration',
            'Detoxification',
            'Mood enhancement',
            'Xeronine for cellular health',
            'Adaptogenic properties'
        ],
        'how_to_use': [
            'Drink noni juice on empty stomach',
            'Apply noni pulp on skin for conditions',
            'Use noni capsules as supplements',
            'Noni leaf tea for internal health',
            'Fermented noni juice',
            'In smoothies (mask taste)',
            'Noni powder with water',
            'Traditional preparations',
            'Noni fruit powder',
            'Noni extract'
        ],
        'precautions': [
            'May interact with blood pressure medications',
            'Can affect liver enzymes - monitor with liver conditions',
            'High potassium content - caution with kidney problems',
            'Start with small doses to check tolerance',
            'Strong taste may cause nausea',
            'Avoid during pregnancy',
            'Consult doctor for medicinal use',
            'May cause digestive issues'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Tikta, Katu (Bitter, Pungent)',
            'Guna (Quality)': 'Laghu, Ruksha (Light, Dry)',
            'Virya (Potency)': 'Ushna (Heating)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Kapha and Vata',
            'Prabhava (Special)': 'Rasayana (Rejuvenative), Vedanasthapana (Analgesic)'
        }
    },
    'oleander': {
        'common_name': 'Oleander (Arali)',
        'scientific_name': 'Nerium oleander',
        'hindi_name': 'कनेर',
        'family': 'Apocynaceae',
        'description': 'Oleander is a beautiful but highly toxic plant used in traditional medicine with extreme caution. All parts are poisonous. A plant of beauty and danger.',
        'medicinal_uses': [
            'Traditional use for skin diseases',
            'Used in heart conditions under expert supervision',
            'Anti-cancer properties being researched',
            'External application for skin problems',
            'Cardiac glycosides for heart',
            'Anti-inflammatory',
            'Antimicrobial',
            'Traditional preparations',
            'Diuretic (traditional)',
            'Emetic (traditional)'
        ],
        'health_benefits': [
            'Cardiac glycosides for heart conditions',
            'Anti-inflammatory properties',
            'Antibacterial effects',
            'Potential anti-cancer compounds',
            'Traditional medicine uses',
            'Skin conditions',
            'Research ongoing',
            'Anti-arrhythmic properties',
            'Cytotoxic effects',
            'Anti-parasitic'
        ],
        'how_to_use': [
            '⚠️ STRICTLY UNDER EXPERT SUPERVISION ONLY',
            '⚠️ Never consume without proper processing',
            '⚠️ External applications only with guidance',
            '⚠️ Traditional formulations by qualified practitioners',
            'Not for home use',
            'Only in classical medicines',
            'Purified forms in Ayurveda',
            'Extreme caution required',
            'Homeopathic preparations',
            'Never self-medicate'
        ],
        'precautions': [
            '⚠️ HIGHLY TOXIC - Can be fatal if ingested',
            '⚠️ Never use without expert guidance',
            '⚠️ Keep away from children and pets',
            '⚠️ Do not self-medicate under any circumstances',
            'All parts are poisonous',
            'Even smoke is toxic',
            'Medical emergency if ingested',
            'Cardiotoxic - affects heart rhythm'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Tikta, Katu (Bitter, Pungent)',
            'Guna (Quality)': 'Laghu, Ruksha, Tikshna (Light, Dry, Sharp)',
            'Virya (Potency)': 'Ushna (Heating)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Kapha and Vata',
            'Prabhava (Special)': 'Hridya (Cardiac tonic - in micro doses), Vishaghna (Anti-venom)'
        }
    },
    'betel_nut': {
        'common_name': 'Betel Nut (Areca Nut)',
        'scientific_name': 'Areca catechu',
        'hindi_name': 'सुपारी',
        'family': 'Arecaceae',
        'description': 'Betel nut has traditional medicinal uses but is known for its stimulant properties and significant health risks. Use with extreme caution. A nut with a dark side.',
        'medicinal_uses': [
            'Traditional digestive aid',
            'Mild stimulant properties',
            'Astringent for oral health',
            'Traditional worm treatment',
            'Skin conditions in small amounts',
            'Diarrhea treatment',
            'Traditional medicine uses',
            'Ayurvedic formulations',
            'Dental health (controversial)',
            'Appetite suppressant'
        ],
        'health_benefits': [
            'Mild stimulant effect',
            'Astringent properties',
            'Traditional digestive aid',
            'Antimicrobial effects in small doses',
            'Oral health (controversial)',
            'Traditional uses',
            'Research ongoing',
            'Cholinergic effects',
            'Anti-depressant (research)',
            'Anti-parasitic'
        ],
        'how_to_use': [
            '⚠️ STRICTLY LIMITED MEDICINAL USE ONLY',
            '⚠️ Traditional formulations under guidance',
            '⚠️ Small amounts for digestive issues',
            '⚠️ External applications only',
            'Not for regular use',
            'Avoid with tobacco',
            'Only in classical medicines',
            'Consult expert',
            'Never for recreational use',
            'Avoid long-term use'
        ],
        'precautions': [
            '⚠️ Known carcinogen with long-term use',
            '⚠️ Highly addictive substance',
            '⚠️ Increases risk of oral cancer',
            '⚠️ Avoid regular consumption',
            '⚠️ Not recommended for medicinal use',
            'Causes oral submucous fibrosis',
            'Dental problems',
            'Addiction risk',
            'Cardiovascular effects',
            'Withdrawal symptoms'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Kashaya, Madhura (Astringent, Sweet)',
            'Guna (Quality)': 'Laghu, Ruksha (Light, Dry)',
            'Virya (Potency)': 'Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Kapha, increases Pitta',
            'Prabhava (Special)': 'Krimighna (Anti-helminthic), Stambhana (Anti-diarrheal)'
        }
    },
    'geranium': {
        'common_name': 'Geranium',
        'scientific_name': 'Pelargonium graveolens',
        'hindi_name': 'गेरेनियम',
        'family': 'Geraniaceae',
        'description': 'Geranium is known for its aromatic leaves and essential oil with therapeutic properties. It is used in aromatherapy and skincare. The "Rose-scented" geranium for beauty and health.',
        'medicinal_uses': [
            'Skin conditions and acne treatment',
            'Stress and anxiety relief',
            'Anti-inflammatory for wounds',
            'Hormonal balance support',
            'Respiratory issues relief',
            'Pain relief',
            'Antimicrobial',
            'Insect repellent',
            'Menstrual problems',
            'Nerve pain'
        ],
        'health_benefits': [
            'Antimicrobial properties',
            'Anti-inflammatory effects',
            'Astringent qualities',
            'Antidepressant properties',
            'Antiseptic for wounds',
            'Hormonal balance',
            'Skin healing',
            'Relaxation',
            'Circulation improvement',
            'Lymphatic stimulation'
        ],
        'how_to_use': [
            'Geranium essential oil for aromatherapy',
            'Leaf paste for skin conditions',
            'Tea for stress relief',
            'Steam inhalation for respiratory issues',
            'Diluted oil for massage',
            'In skincare products',
            'Potpourri for fragrance',
            'Compress for pain',
            'Geranium water as toner',
            'Bath oil for relaxation'
        ],
        'precautions': [
            'Essential oil should be diluted',
            'May cause skin irritation in some',
            'Avoid during pregnancy',
            'Patch test before skin application',
            'Not for internal use without guidance',
            'Consult for medicinal use',
            'Keep away from children',
            'May interact with medications'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Kashaya, Tikta (Astringent, Bitter)',
            'Guna (Quality)': 'Laghu, Snigdha (Light, Unctuous)',
            'Virya (Potency)': 'Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Katu (Pungent)',
            'Dosha Effect': 'Balances Pitta and Kapha',
            'Prabhava (Special)': 'Varnya (Improves complexion), Vedanasthapana (Analgesic)'
        }
    },
    'wood_sorrel': {
        'common_name': 'Wood Sorrel',
        'scientific_name': 'Oxalis acetosella',
        'hindi_name': 'खट्टी बूटी',
        'family': 'Oxalidaceae',
        'description': 'Wood Sorrel is a small medicinal plant with sour taste, known for its cooling and digestive properties. It is rich in Vitamin C. The "Sour Grass" of the forests.',
        'medicinal_uses': [
            'Fever and inflammation reduction',
            'Digestive issues and appetite improvement',
            'Skin conditions and wounds',
            'Urinary tract infections',
            'Mouth ulcers and sore throat',
            'Scurvy prevention',
            'Detoxification',
            'Cooling effect',
            'Anti-helminthic',
            'Diuretic'
        ],
        'health_benefits': [
            'Rich in Vitamin C',
            'Antioxidant properties',
            'Anti-inflammatory effects',
            'Diuretic properties',
            'Cooling effect on body',
            'Digestive stimulant',
            'Blood purification',
            'Immune support',
            'Anti-microbial',
            'Anti-scorbutic'
        ],
        'how_to_use': [
            'Chew leaves for digestive issues',
            'Leaf juice for fever',
            'Paste application for skin',
            'Herbal tea for urinary problems',
            'In salads for sour taste',
            'Chutney for digestion',
            'Infusion for mouth ulcers',
            'Traditional preparations',
            'Leaf poultice for wounds',
            'Sorrel soup'
        ],
        'precautions': [
            'Contains oxalic acid - avoid in large quantities',
            'May interact with kidney medications',
            'Not recommended for people with kidney stones',
            'Use in moderation',
            'Avoid during pregnancy',
            'May cause allergies in some',
            'Consult for medicinal use',
            'May cause calcium deficiency'
        ],
        'ayurvedic_properties': {
            'Rasa (Taste)': 'Amla (Sour)',
            'Guna (Quality)': 'Laghu, Snigdha (Light, Unctuous)',
            'Virya (Potency)': 'Sheeta (Cooling)',
            'Vipaka (Post-digestive)': 'Amla (Sour)',
            'Dosha Effect': 'Balances Pitta and Kapha',
            'Prabhava (Special)': 'Deepana (Digestive), Rochana (Appetizer)'
        }
    },
    'unknown': {
        'common_name': 'Unknown Plant',
        'scientific_name': 'Unidentified Species',
        'hindi_name': 'अपरिचित पौधा',
        'family': 'Unknown Family',
        'description': 'This plant could not be identified with sufficient confidence. It may not be in our medicinal plants database or the image quality may be insufficient. Please consult an expert for proper identification.',
        'ayurvedic_properties': {
            'Rasa': 'Unknown - Requires proper identification',
            'Guna': 'Unknown',
            'Virya': 'Unknown',
            'Vipaka': 'Unknown',
            'Dosha Effect': 'Unknown',
            'Prabhava': 'Unknown'
        },
        'medicinal_uses': [
            'Cannot recommend medicinal uses for unidentified plants',
            'Consult with a botanist or Ayurvedic expert',
            'Proper identification is essential for safe usage',
            'Do not use without expert verification',
            'Some plants may be toxic if misidentified'
        ],
        'health_benefits': [
            'Unknown - requires proper identification',
            'Some plants may be toxic if misidentified',
            'Always verify with experts before use',
            'Safety first - do not experiment',
            'Consult multiple sources for identification'
        ],
        'how_to_use': [
            '⚠️ DO NOT USE until properly identified',
            '⚠️ Consult local botanical garden or expert',
            '⚠️ Take clear photos from multiple angles for identification',
            '⚠️ Some plants can be toxic or poisonous',
            '⚠️ Never consume unidentified plants',
            '⚠️ Keep away from children and pets'
        ],
        'precautions': [
            '⚠️ DO NOT CONSUME unidentified plants',
            '⚠️ Some plants can be toxic or poisonous',
            '⚠️ Always verify with multiple sources',
            '⚠️ Consult qualified Ayurvedic practitioner',
            '⚠️ Keep away from children and pets',
            '⚠️ Seek expert help for identification',
            '⚠️ When in doubt, throw it out'
        ]
    }
}

# Transform the database to match the expected format for the app
PLANTS_INFO = {}
for plant_key, plant_data in medicinal_plants_database.items():
    # Format the plant name properly (capitalize first letter)
    formatted_name = plant_key.replace('_', ' ').title()
    
    # Extract the relevant fields and map to the expected structure
    PLANTS_INFO[formatted_name] = {
        'scientific_name': plant_data.get('scientific_name', 'N/A'),
        'family': plant_data.get('family', 'N/A'),
        'common_names': [plant_data.get('common_name', '')],
        'hindi_name': plant_data.get('hindi_name', ''),
        'ayurvedic_name': plant_data.get('ayurvedic_properties', {}).get('Prabhava (Special)', 'N/A'),
        'description': plant_data.get('description', ''),
        'uses_real_life': plant_data.get('medicinal_uses', []),
        'health_benefits': plant_data.get('health_benefits', []),
        'ayurvedic_uses': plant_data.get('ayurvedic_uses', plant_data.get('medicinal_uses', [])[:5]),  # Use first 5 medicinal uses if ayurvedic_uses not present
        'preparation_methods': plant_data.get('how_to_use', []),
        'precautions': ' '.join(plant_data.get('precautions', [])) if plant_data.get('precautions') else 'No specific precautions listed.',
        'ayurvedic_properties': plant_data.get('ayurvedic_properties', {}),
        'image_url': f'/static/images/{plant_key}.jpg'
    }

# Also add entries for all possible plant name variations that might come from the model
# This ensures that whatever naming convention the model uses, we have data available
for plant_key in medicinal_plants_database.keys():
    # Add the raw key (like 'tulsi')
    PLANTS_INFO[plant_key] = PLANTS_INFO.get(plant_key.replace('_', ' ').title(), PLANTS_INFO.get('Unknown'))
    
    # Add title case with spaces (like 'Tulsi')
    title_case = plant_key.replace('_', ' ').title()
    PLANTS_INFO[title_case] = PLANTS_INFO.get(title_case, PLANTS_INFO.get('Unknown'))

print(f"Loaded {len(PLANTS_INFO)} plants into the database")

# Load the model and class indices
print("Loading model...")
try:
    model = keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please check if the model file exists at the specified path")
    model = None

try:
    with open(CLASS_INDICES_PATH, 'rb') as f:
        class_indices = pickle.load(f)
    print("Class indices loaded successfully!")
    
    # Reverse the class indices to get class names from indices
    idx_to_class = {v: k for k, v in class_indices.items()}
    class_names = list(class_indices.keys())
    print(f"Total classes: {len(class_names)}")
    print(f"Classes: {class_names}")
except FileNotFoundError:
    print(f"Error: Class indices file not found at {CLASS_INDICES_PATH}")
    print("Using fallback class names from PLANTS_INFO")
    # Create a fallback class list based on your PLANTS_INFO
    class_names = list(medicinal_plants_database.keys())
    idx_to_class = {i: name for i, name in enumerate(class_names)}
    class_indices = {name: i for i, name in enumerate(class_names)}
    print(f"Using fallback class names from medicinal_plants_database: {len(class_names)} classes")
    print(f"Classes: {class_names}")

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_plant(img_path):
    """Predict plant from image"""
    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize
    
    # Make prediction
    predictions = model.predict(img_array, verbose=0)
    predicted_class_idx = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_class_idx])
    
    # Get class name
    predicted_class = idx_to_class[predicted_class_idx]
    
    # Get top 3 predictions
    top_3_idx = np.argsort(predictions[0])[-3:][::-1]
    top_3_predictions = [
        {
            'class': idx_to_class[idx],
            'confidence': float(predictions[0][idx])
        }
        for idx in top_3_idx
    ]
    
    return predicted_class, confidence, top_3_predictions

@app.route('/')
def index():
    return render_template('index.html', plants=class_names)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        predicted_class, confidence, top_3 = predict_plant(filepath)
        
        # Get plant information - try different name formats
        plant_info = None
        
        # Try exact match
        if predicted_class in PLANTS_INFO:
            plant_info = PLANTS_INFO[predicted_class]
        else:
            # Try title case
            title_case = predicted_class.replace('_', ' ').title()
            if title_case in PLANTS_INFO:
                plant_info = PLANTS_INFO[title_case]
            else:
                # Try lower case
                lower_case = predicted_class.lower()
                if lower_case in PLANTS_INFO:
                    plant_info = PLANTS_INFO[lower_case]
                else:
                    # Try to find in medicinal_plants_database directly
                    for key in medicinal_plants_database.keys():
                        if key.lower() == predicted_class.lower() or key.replace('_', ' ').lower() == predicted_class.lower():
                            # Convert database entry to PLANTS_INFO format
                            plant_data = medicinal_plants_database[key]
                            plant_info = {
                                'scientific_name': plant_data.get('scientific_name', 'N/A'),
                                'family': plant_data.get('family', 'N/A'),
                                'common_names': [plant_data.get('common_name', '')],
                                'hindi_name': plant_data.get('hindi_name', ''),
                                'description': plant_data.get('description', ''),
                                'uses_real_life': plant_data.get('medicinal_uses', []),
                                'health_benefits': plant_data.get('health_benefits', []),
                                'preparation_methods': plant_data.get('how_to_use', []),
                                'precautions': ' '.join(plant_data.get('precautions', [])) if plant_data.get('precautions') else 'No specific precautions listed.',
                                'ayurvedic_properties': plant_data.get('ayurvedic_properties', {})
                            }
                            break
                    
                    # If still not found, use unknown
                    if not plant_info:
                        plant_info = PLANTS_INFO.get('Unknown', {
                            'description': 'Information not available',
                            'uses_real_life': [],
                            'scientific_name': 'N/A',
                            'family': 'N/A',
                            'common_names': [],
                            'hindi_name': '',
                            'preparation_methods': [],
                            'precautions': 'No precautionary information available.'
                        })
        
        # Create image URL for display
        image_url = url_for('static', filename=f'uploads/{filename}')
        
        return jsonify({
            'success': True,
            'predicted_class': predicted_class,
            'confidence': confidence,
            'top_3_predictions': top_3,
            'image_url': image_url,
            'plant_info': plant_info
        })
    
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/plant/<plant_name>')
def plant_detail(plant_name):
    # Try different formats to find the plant
    plant_info = None
    
    # Try exact match
    if plant_name in PLANTS_INFO:
        plant_info = PLANTS_INFO[plant_name]
    else:
        # Try title case
        title_case = plant_name.replace('_', ' ').title()
        if title_case in PLANTS_INFO:
            plant_info = PLANTS_INFO[title_case]
        else:
            # Try lower case
            lower_case = plant_name.lower()
            if lower_case in PLANTS_INFO:
                plant_info = PLANTS_INFO[lower_case]
            else:
                # Try to find in medicinal_plants_database directly
                for key in medicinal_plants_database.keys():
                    if key.lower() == plant_name.lower() or key.replace('_', ' ').lower() == plant_name.lower():
                        # Convert database entry to PLANTS_INFO format
                        plant_data = medicinal_plants_database[key]
                        plant_info = {
                            'scientific_name': plant_data.get('scientific_name', 'N/A'),
                            'family': plant_data.get('family', 'N/A'),
                            'common_names': [plant_data.get('common_name', '')],
                            'hindi_name': plant_data.get('hindi_name', ''),
                            'description': plant_data.get('description', ''),
                            'uses_real_life': plant_data.get('medicinal_uses', []),
                            'health_benefits': plant_data.get('health_benefits', []),
                            'preparation_methods': plant_data.get('how_to_use', []),
                            'precautions': ' '.join(plant_data.get('precautions', [])) if plant_data.get('precautions') else 'No specific precautions listed.',
                            'ayurvedic_properties': plant_data.get('ayurvedic_properties', {})
                        }
                        break
    
    if plant_info:
        return render_template('plant_detail.html', plant_name=plant_name, plant_info=plant_info)
    else:
        return "Plant information not found", 404

@app.route('/api/plants')
def get_plants():
    """API endpoint to get list of all plants"""
    plants_list = []
    for plant_name in medicinal_plants_database.keys():
        info = medicinal_plants_database.get(plant_name, {})
        plants_list.append({
            'name': plant_name.replace('_', ' ').title(),
            'scientific_name': info.get('scientific_name', ''),
            'common_name': info.get('common_name', ''),
            'hindi_name': info.get('hindi_name', ''),
            'description': info.get('description', '')[:100] + '...' if info.get('description') else ''
        })
    return jsonify(plants_list)

@app.route('/api/plant/<plant_name>')
def get_plant_info(plant_name):
    """API endpoint to get detailed plant information"""
    # Try to find in medicinal_plants_database directly
    plant_info = None
    for key in medicinal_plants_database.keys():
        if key.lower() == plant_name.lower() or key.replace('_', ' ').lower() == plant_name.lower():
            plant_info = medicinal_plants_database[key]
            break
    
    if plant_info:
        return jsonify(plant_info)
    else:
        return jsonify({'error': 'Plant not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)