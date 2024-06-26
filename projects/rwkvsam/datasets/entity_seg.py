from mmdet.datasets.coco import CocoDataset
from mmdet.registry import DATASETS

THING_CLASSES = (
    'table_ot',
    'rigid_container_ot',
    'box',
    'cup_ot',
    'cable',
    # 'unknown',
    'chair',
    'poster',
    'drink_ot',
    'tissue',
    'towel',
    'plate',
    'gamepad',
    'kitchen cabinets',
    'rug',
    'paper',
    'clean_tool_ot',
    'kitchen_sink',
    'man',
    'facility_ot',
    'can_light',
    'kitchen_pot_ot',
    'dining_table',
    'cow_ot',
    'signboard',
    'motorbike',
    'basket',
    'woman',
    'barrel',
    'electric_wire',
    'handbag',
    'bag_ot',
    'painting',
    'cushion',
    'mat',
    'utility_ot',
    'plugs_and_sockets',
    'stool',
    'couch',
    'noncommon_furniture',
    'ceiling_lamp',
    'ordniary_sofa',
    'blanket',
    'remote_control',
    'media_ot',
    'flexible_container_ot',
    'fooddrink_ot',
    'doll_ot',
    'hat_ot',
    'cake',
    'trash_bin',
    'umbrella',
    'person_ot',
    'outdoor_supplies_ot',
    'cigarette',
    'microphone',
    'motorboat',
    'street_light',
    'tv_receiver',
    'car',
    'bicycle',
    'boy',
    'land_transportation_ot',
    'mobilephone',
    'pen',
    'laptop',
    'chandilier',
    'entertainment_appliances_ot',
    'plastic_bag',
    'wall clock',
    'coat',
    'surveillance_camera',
    'blackboard',
    'cloth_ot',
    'billboard',
    'horse_ot',
    'pickup_truck',
    'rope',
    'baseball_cap',
    'backpack',
    'book',
    'sofa_ot',
    'girl',
    'hambuger',
    'hose',
    'filling cabinets',
    'chest_of_drawers',
    'footwear_ot',
    'giraffe',
    'branch',
    'vegetable_ot',
    'gas_stove',
    'corded_telephone',
    'kitchen_tool_ot',
    'kettle',
    'cutting_board',
    'light_ot',
    'ventilation_appliances_ot',
    'kitchen_appliances_ot',
    'glass',
    'fork',
    'napkin',
    'candle',
    'pie',
    'curtain',
    'traffic_sign',
    'traffic_light',
    'clock',
    'spoon',
    'photo',
    'wineglass',
    'tableware',
    'table-knife',
    'double_sofa',
    'bread',
    'platter',
    'scooter',
    'bedroom bed',
    'pillow',
    'splippers',
    'nightstand',
    'television',
    'air_conditioner',
    'bus',
    'integrated_table_and_chair',
    'teddy_bear',
    'bowl',
    'mouse',
    'speaker',
    'keyboard',
    'screen',
    'desk',
    'paper_bag',
    'alarm clock',
    'mango',
    'orange',
    'flower',
    'leaf',
    'food_processor',
    'wardrobe',
    'toy_ot',
    'digital clock',
    'seat_furniture_ot',
    'baseball bat',
    'baseball',
    'dance_pole',
    'bottle',
    'saucer',
    'hanger',
    'racket_ot',
    'flag',
    'camera',
    'cabinets_ot',
    'chopsticks',
    'packaging_paper',
    'switch',
    ' folder',
    'apple',
    'trunk',
    'sculpture',
    'suitcase',
    'guideboard',
    'pizza',
    'sconce',
    'ceiling_fan',
    'earthware_pot_with_handle',
    'kite',
    'condiment',
    'fast_food_ot',
    'decoration',
    'teapot',
    'pineapple',
    'bulletin_board',
    'mirror',
    'briefcase',
    'cellular_telephone',
    'radiator',
    'beddings_ot',
    'frying_pan',
    'range_hood',
    'hammer',
    'coffeemaker',
    'micro-wave_oven',
    'banana',
    'oven',
    'appliances_ot',
    'sika_deer',
    'van',
    'tennis_racket',
    'upper_body_clothing_os',
    'dog',
    'tennis',
    'fitness_equipment_ot',
    'banner',
    'frisbee',
    'train',
    'kiosk',
    'meat_ot',
    'helmet_ot',
    'jar',
    'fruit_ot',
    'football',
    'can',
    'drinking_straw',
    'ballon',
    'T-shirt',
    'sword',
    'binoculars',
    'mechanical_fan',
    'accessories_ot',
    'knife_ot',
    'bagel',
    'bench',
    'vent',
    'ship',
    'blanket_furniture_ot',
    'toothbrush',
    'table_cloth',
    'bicyclehelmet',
    'flowerpot',
    'dessert_snacks_ot',
    'sheep',
    'crane',
    'rugby',
    'truck_ot',
    'surfboard',
    'slow_cooker',
    'spatula',
    'dishwasher',
    'pigeon',
    'canary',
    'birds_ot',
    'mattress',
    'bed_ot',
    'crutch',
    'radio',
    'leather_shoes',
    'sneakers',
    'tomato',
    'electronic_stove',
    'doughnut',
    'fire_hyrant',
    'musical_instrument',
    'carriage',
    'airplane',
    'salt_shaker',
    'repair_tool_ot',
    'paddle',
    'boat',
    'ski',
    'ski_pole',
    'gloves',
    'hotdog',
    'french_fries',
    'trolley',
    'wood_stove',
    'sausage',
    'refrigerator',
    'telephone_ot',
    'coffee cup',
    'hoodie',
    'elephant',
    'zebra',
    'blanket',
    'converter',
    'cat',
    'played_blocks',
    'segway',
    'sunhat',
    'extension_cord',
    'ladder',
    'scarf',
    'faucet',
    'plant_ot',
    'wine',
    'goggles',
    'measuring cup',
    'juice',
    'wheelchair',
    'pipe',
    'clock_furniture_ot',
    'grape',
    'toaster',
    'lightbulb',
    'seasoning_ot',
    'balance',
    'watermelon',
    'salad',
    'tire',
    'ball_ot',
    'gym_equipment',
    'liquid_soap',
    'bell_papper',
    'container_ot',
    'toilet_paper',
    'kitchen-knife',
    'cake_stand',
    'soap',
    'bath_tool_ot',
    'toilet',
    'sunglasses',
    'parking_meter',
    'watch',
    'vehicle_part_ot',
    'ski_board',
    'deer_ot',
    'mammal_ot',
    'cage',
    'monkey',
    'sunglasses_ot',
    'sealion',
    'tank',
    'eagle',
    'camel_ot',
    'pomegranate',
    'plant_part_ot',
    'lemon',
    'duck',
    'canoe',
    ' high-heeled_shoes',
    'cookie',
    'leopard',
    'fox',
    'seal',
    'peach',
    'rabbit',
    'beetle',
    'solar_pannel',
    'pumkin',
    'hot_ballon',
    'turtle',
    'fish_ot',
    'pig',
    'jellyfish',
    'lizard',
    'crab',
    'squirrel',
    'polar_bear',
    'spaghetti',
    'dog bed',
    'shampoo',
    'bathhub',
    'storage box',
    'one-piece_dress',
    'sweing_machine',
    'lab_tool_ot',
    'pants',
    'shirt',
    'human_accessories_ot',
    'strawberry',
    'papper',
    'helicopter',
    'sailboat',
    'necklace',
    'lion',
    'billards',
    'billiard_table',
    'volleyball',
    'conveyor_belt',
    'weapon_ot',
    'tennis_table',
    'stove_ot',
    'ruler',
    'tube',
    'beaker',
    'coffee_table',
    'windshield_wiper',
    'scoreboard',
    'mast',
    'pressure_cooker',
    'broom',
    'potato',
    'carrot',
    'vacuum_cleaner',
    'mop',
    'bathroom cabinets',
    'sandwich',
    'mixer',
    'brown_bear',
    'slide',
    'waffle',
    'pear',
    'alpaca',
    'dice',
    'poker',
    'parachute',
    'basketball',
    'parrot',
    'taco',
    'golf',
    'golf_cart',
    'ice_cream',
    'pretzel',
    'cabbage',
    'broccolli',
    'hairdryer',
    'zucchini',
    'ax',
    'mushroom',
    'fishing_tool',
    'dress_ot',
    'goldfish',
    'shark',
    'chicken',
    'sparrow',
    'owl',
    'amphibians_ot',
    'drug',
    'frog',
    'missle',
    'crocodile',
    'snake',
    'insect_ot',
    'spider',
    'reptile_ot',
    'shovel',
    'parasite',
    'centipede',
    'peacock',
    'rifle',
    'goose',
    'kangaroo',
    'mollusca_ot',
    'crustacea_ot',
    'snail',
    'lobster',
    'drill',
    'animal_ot',
    'spacecraft',
    'punching_bag',
    'handgun',
    'whale',
    'brush',
    'screwdriver',
    'snowmobile',
    'washing_machine',
    'vehicle_ot',
    'stretcher',
    'trophies_medals_certification',
    'swing',
    'tank',
    'barbie_doll',
    'tiger',
    'underwater_vehicle_ot',
    'dinosar',
    'bear_ot',
    'hamster',
    'ladybug',
    'bee',
    'dragonfly',
    'moth',
    'starfish',
    'antelope',
    'raccoon',
    'dumbbell',
    'crib',
    'bow_and_arrow',
    'cannon',
    'stuffed-animal',
    'hose',
    'snow_plow',
    'tripod',
    'coffee',
    'radish',
    'caterpillar',
    'puff_pastry',
    'woodpecker',
    'sandals',
    'popcorn',
    'wrench',
    'flashlight',
    'earrings',
    ' medical_equipment',
    'egg',
    'cucumber',
    'ring',
    'tong',
    'candy',
    'tank_top',
    'seafood_ot',
    'ant',
    'onion',
    'lower_body_clothing_ot',
    'beehive',
    'jewlery_ot',
    'treadmill',
    'trousers_ot',
    'crossbar',
    'shrimp',
    'seahorse',
    'rhinoceros',
    'lantern',
    'commonfig',
    'golf_club',
    'drawer',
    'dagger',
    'table_tennis_racket',
    'casserole',
    'sushi',
    'fitness_chair',
    'blueberry',
    'cocktail',
    'pancake',
    'crumb',
    'nail_clippers',
    'clean_appliances_ot',
    'asparagus',
    'coconut',
    'jacket',
    'hammock',
    'wheel',
    'dolphin',
    'cable_car',
    'chess',
    'boat_part_ot',
    'grapefruit',
    'crow',
    'bracelet',
    'air_traffic_ot',
    'donkey',
    'cowboy_hat',
    'crown',
    'dustpan',
    'jersey',
    'bolt',
    'sofa_bed',
    'sports_musical_ot',
    'cradle',
    'porcupine',
    'sleeping-bag',
    'shorts',
    'hockey',
    'shotgun',
    'wax_gourd',
    'rearview_mirror',
)

STUFF_CLASSES = (
    'unknown',
    'floor',
    'wall_ot',
    'park_ground',
    'banister',
    'house',
    'bar',
    'shrub',
    'step',
    'potted_plants',
    'door',
    'ceiling',
    'window',
    'building_houses_ot',
    'sky',
    'electricpole',
    'lake',
    'grass',
    'sidewalk',
    'tree',
    'fence',
    'non-buildinghouse_ot',
    'sunk_fence',
    'stone',
    'rock',
    'road',
    'glass_window',
    'blur',
    'high-rise',
    'barrier_ot',
    'vine',
    'path',
    'pipe',
    'building_structure_ot',
    'court',
    'swimmingpool',
    'concretewall',
    'fountain',
    'soil',
    'wiredfence',
    'glasswall',
    'artifical_ground_ot',
    'tent',
    'sea',
    'ventilation',
    'mountain',
    'log',
    'building_ot',
    'floor_structure_ot',
    'gutter',
    'sand',
    'seating_area',
    'playground',
    'railroad',
    'platform',
    'christmas_tree',
    'pole',
    'fireplace',
    'straw',
    'stair',
    'nonindividual_plants_ot',
    'fluid_ot',
    'river',
    'snow',
    'dirt',
    'trail',
    'pond',
    'gravel',
    'church',
    'countertop',
    'fountain_ground',
    'parterre',
    'chimney',
    'stage',
    'awning',
    'crops',
    'statium',
    'alley',
    'moss',
    'rose',
    'moon',
    'tower',
    'pier',
    'parkinglot',
    'bridge',
    'palm',
    'individual_plants_ot',
    'waterfall',
    'castle',
    'stove',
    'temple',
    'escalator',
    'sun',
    'solid_ot',
    'skyscraper',
    'tunnel',
    'brook',
    'cement_floor',
    'waterdrop',
    'inanimate_natural_objects_ot',
    'cage',
    'balcony',
    'sunflower',
    'lighthouse',
    'rainbow',
    'cloud',
    'lily',
    'lavender',
    'maple',
    'willow',
)

_CLASSES = (*THING_CLASSES, *STUFF_CLASSES)
CLASSES = (
    'table_ot',
    'rigid_container_ot',
    'box',
    'cup_ot',
    'cable',
    'unknown',
    'chair',
    'poster',
    'drink_ot',
    'tissue',
    'towel',
    'plate',
    'gamepad',
    'kitchen cabinets',
    'rug',
    'floor',
    'paper',
    'clean_tool_ot',
    'kitchen_sink',
    'man',
    'facility_ot',
    'wall_ot',
    'can_light',
    'kitchen_pot_ot',
    'dining_table',
    'park_ground',
    'cow_ot',
    'signboard',
    'motorbike',
    'basket',
    'banister',
    'woman',
    'house',
    'barrel',
    'bar',
    'electric_wire',
    'handbag',
    'bag_ot',
    'shrub',
    'step',
    'painting',
    'potted_plants',
    'cushion',
    'mat',
    'utility_ot',
    'plugs_and_sockets',
    'stool',
    'couch',
    'noncommon_furniture',
    'ceiling_lamp',
    'ordniary_sofa',
    'door',
    'blanket',
    'ceiling',
    'remote_control',
    'media_ot',
    'window',
    'flexible_container_ot',
    'fooddrink_ot',
    'doll_ot',
    'hat_ot',
    'cake',
    'trash_bin',
    'umbrella',
    'person_ot',
    'building_houses_ot',
    'outdoor_supplies_ot',
    'sky',
    'cigarette',
    'microphone',
    'motorboat',
    'street_light',
    'electricpole',
    'lake',
    'grass',
    'tv_receiver',
    'car',
    'sidewalk',
    'tree',
    'bicycle',
    'fence',
    'boy',
    'non-buildinghouse_ot',
    'sunk_fence',
    'land_transportation_ot',
    'mobilephone',
    'pen',
    'laptop',
    'chandilier',
    'entertainment_appliances_ot',
    'plastic_bag',
    'wall clock',
    'coat',
    'surveillance_camera',
    'blackboard',
    'cloth_ot',
    'billboard',
    'horse_ot',
    'pickup_truck',
    'rope',
    'baseball_cap',
    'backpack',
    'book',
    'sofa_ot',
    'girl',
    'hambuger',
    'hose',
    'filling cabinets',
    'chest_of_drawers',
    'footwear_ot',
    'stone',
    'giraffe',
    'rock',
    'branch',
    'vegetable_ot',
    'gas_stove',
    'corded_telephone',
    'kitchen_tool_ot',
    'kettle',
    'cutting_board',
    'light_ot',
    'ventilation_appliances_ot',
    'kitchen_appliances_ot',
    'glass',
    'fork',
    'napkin',
    'candle',
    'pie',
    'curtain',
    'traffic_sign',
    'traffic_light',
    'clock',
    'road',
    'spoon',
    'photo',
    'wineglass',
    'tableware',
    'table-knife',
    'double_sofa',
    'glass_window',
    'bread',
    'blur',
    'platter',
    'scooter',
    'high-rise',
    'bedroom bed',
    'pillow',
    'splippers',
    'nightstand',
    'television',
    'air_conditioner',
    'barrier_ot',
    'vine',
    'bus',
    'integrated_table_and_chair',
    'teddy_bear',
    'bowl',
    'path',
    'mouse',
    'speaker',
    'keyboard',
    'screen',
    'desk',
    'paper_bag',
    'alarm clock',
    'mango',
    'orange',
    'flower',
    'leaf',
    'food_processor',
    'wardrobe',
    'pipe',
    'toy_ot',
    'digital clock',
    'building_structure_ot',
    'court',
    'seat_furniture_ot',
    'baseball bat',
    'baseball',
    'dance_pole',
    'bottle',
    'swimmingpool',
    'saucer',
    'hanger',
    'racket_ot',
    'flag',
    'camera',
    'cabinets_ot',
    'chopsticks',
    'packaging_paper',
    'switch',
    ' folder',
    'concretewall',
    'apple',
    'trunk',
    'fountain',
    'soil',
    'sculpture',
    'wiredfence',
    'suitcase',
    'guideboard',
    'pizza',
    'glasswall',
    'sconce',
    'ceiling_fan',
    'earthware_pot_with_handle',
    'kite',
    'artifical_ground_ot',
    'tent',
    'condiment',
    'fast_food_ot',
    'decoration',
    'teapot',
    'sea',
    'pineapple',
    'ventilation',
    'mountain',
    'bulletin_board',
    'mirror',
    'log',
    'briefcase',
    'cellular_telephone',
    'radiator',
    'beddings_ot',
    'frying_pan',
    'range_hood',
    'hammer',
    'coffeemaker',
    'micro-wave_oven',
    'banana',
    'oven',
    'appliances_ot',
    'sika_deer',
    'building_ot',
    'van',
    'floor_structure_ot',
    'gutter',
    'sand',
    'tennis_racket',
    'upper_body_clothing_os',
    'seating_area',
    'playground',
    'dog',
    'tennis',
    'fitness_equipment_ot',
    'banner',
    'frisbee',
    'railroad',
    'platform',
    'train',
    'kiosk',
    'meat_ot',
    'christmas_tree',
    'pole',
    'helmet_ot',
    'fireplace',
    'jar',
    'fruit_ot',
    'football',
    'can',
    'drinking_straw',
    'ballon',
    'T-shirt',
    'straw',
    'sword',
    'binoculars',
    'mechanical_fan',
    'accessories_ot',
    'knife_ot',
    'bagel',
    'bench',
    'stair',
    'vent',
    'ship',
    'blanket_furniture_ot',
    'toothbrush',
    'table_cloth',
    'bicyclehelmet',
    'flowerpot',
    'dessert_snacks_ot',
    'sheep',
    'crane',
    'nonindividual_plants_ot',
    'rugby',
    'truck_ot',
    'surfboard',
    'slow_cooker',
    'spatula',
    'dishwasher',
    'pigeon',
    'canary',
    'birds_ot',
    'fluid_ot',
    'mattress',
    'bed_ot',
    'crutch',
    'radio',
    'leather_shoes',
    'sneakers',
    'tomato',
    'electronic_stove',
    'doughnut',
    'fire_hyrant',
    'musical_instrument',
    'carriage',
    'river',
    'airplane',
    'salt_shaker',
    'repair_tool_ot',
    'paddle',
    'boat',
    'ski',
    'ski_pole',
    'gloves',
    'snow',
    'dirt',
    'hotdog',
    'french_fries',
    'trolley',
    'wood_stove',
    'sausage',
    'refrigerator',
    'telephone_ot',
    'coffee cup',
    'hoodie',
    'elephant',
    'trail',
    'zebra',
    'blanket',
    'pond',
    'converter',
    'cat',
    'played_blocks',
    'segway',
    'sunhat',
    'extension_cord',
    'ladder',
    'scarf',
    'gravel',
    'faucet',
    'church',
    'plant_ot',
    'countertop',
    'wine',
    'goggles',
    'measuring cup',
    'juice',
    'wheelchair',
    'pipe',
    'fountain_ground',
    'clock_furniture_ot',
    'grape',
    'toaster',
    'lightbulb',
    'seasoning_ot',
    'parterre',
    'balance',
    'watermelon',
    'chimney',
    'salad',
    'tire',
    'ball_ot',
    'gym_equipment',
    'liquid_soap',
    'bell_papper',
    'container_ot',
    'toilet_paper',
    'kitchen-knife',
    'cake_stand',
    'stage',
    'awning',
    'soap',
    'bath_tool_ot',
    'toilet',
    'sunglasses',
    'parking_meter',
    'watch',
    'crops',
    'vehicle_part_ot',
    'ski_board',
    'deer_ot',
    'mammal_ot',
    'statium',
    'alley',
    'cage',
    'monkey',
    'sunglasses_ot',
    'sealion',
    'moss',
    'tank',
    'rose',
    'eagle',
    'moon',
    'tower',
    'camel_ot',
    'pomegranate',
    'plant_part_ot',
    'lemon',
    'pier',
    'duck',
    'canoe',
    ' high-heeled_shoes',
    'cookie',
    'leopard',
    'fox',
    'seal',
    'peach',
    'rabbit',
    'beetle',
    'solar_pannel',
    'parkinglot',
    'bridge',
    'palm',
    'pumkin',
    'hot_ballon',
    'individual_plants_ot',
    'turtle',
    'fish_ot',
    'waterfall',
    'pig',
    'jellyfish',
    'lizard',
    'crab',
    'squirrel',
    'polar_bear',
    'castle',
    'spaghetti',
    'stove',
    'dog bed',
    'shampoo',
    'bathhub',
    'storage box',
    'one-piece_dress',
    'sweing_machine',
    'lab_tool_ot',
    'pants',
    'shirt',
    'human_accessories_ot',
    'strawberry',
    'papper',
    'helicopter',
    'sailboat',
    'necklace',
    'lion',
    'billards',
    'billiard_table',
    'volleyball',
    'conveyor_belt',
    'temple',
    'escalator',
    'weapon_ot',
    'tennis_table',
    'stove_ot',
    'sun',
    'ruler',
    'tube',
    'beaker',
    'solid_ot',
    'coffee_table',
    'skyscraper',
    'windshield_wiper',
    'tunnel',
    'scoreboard',
    'mast',
    'pressure_cooker',
    'broom',
    'potato',
    'carrot',
    'vacuum_cleaner',
    'mop',
    'bathroom cabinets',
    'sandwich',
    'mixer',
    'brook',
    'brown_bear',
    'slide',
    'waffle',
    'pear',
    'alpaca',
    'dice',
    'poker',
    'parachute',
    'basketball',
    'parrot',
    'taco',
    'golf',
    'golf_cart',
    'ice_cream',
    'pretzel',
    'cabbage',
    'broccolli',
    'hairdryer',
    'zucchini',
    'ax',
    'mushroom',
    'fishing_tool',
    'dress_ot',
    'goldfish',
    'shark',
    'chicken',
    'sparrow',
    'owl',
    'amphibians_ot',
    'drug',
    'frog',
    'missle',
    'crocodile',
    'snake',
    'insect_ot',
    'spider',
    'reptile_ot',
    'shovel',
    'parasite',
    'centipede',
    'peacock',
    'rifle',
    'goose',
    'cement_floor',
    'kangaroo',
    'mollusca_ot',
    'crustacea_ot',
    'snail',
    'lobster',
    'drill',
    'animal_ot',
    'spacecraft',
    'punching_bag',
    'handgun',
    'whale',
    'brush',
    'screwdriver',
    'snowmobile',
    'washing_machine',
    'vehicle_ot',
    'stretcher',
    'trophies_medals_certification',
    'swing',
    'tank',
    'barbie_doll',
    'waterdrop',
    'tiger',
    'underwater_vehicle_ot',
    'dinosar',
    'bear_ot',
    'hamster',
    'ladybug',
    'bee',
    'dragonfly',
    'moth',
    'starfish',
    'antelope',
    'raccoon',
    'inanimate_natural_objects_ot',
    'dumbbell',
    'crib',
    'bow_and_arrow',
    'cannon',
    'stuffed-animal',
    'hose',
    'snow_plow',
    'tripod',
    'coffee',
    'radish',
    'caterpillar',
    'puff_pastry',
    'woodpecker',
    'sandals',
    'popcorn',
    'wrench',
    'flashlight',
    'earrings',
    ' medical_equipment',
    'cage',
    'egg',
    'cucumber',
    'ring',
    'tong',
    'candy',
    'tank_top',
    'seafood_ot',
    'ant',
    'onion',
    'balcony',
    'sunflower',
    'lower_body_clothing_ot',
    'beehive',
    'jewlery_ot',
    'treadmill',
    'trousers_ot',
    'crossbar',
    'shrimp',
    'seahorse',
    'rhinoceros',
    'lantern',
    'commonfig',
    'golf_club',
    'drawer',
    'dagger',
    'lighthouse',
    'table_tennis_racket',
    'casserole',
    'sushi',
    'fitness_chair',
    'blueberry',
    'cocktail',
    'pancake',
    'crumb',
    'nail_clippers',
    'clean_appliances_ot',
    'asparagus',
    'coconut',
    'jacket',
    'hammock',
    'wheel',
    'dolphin',
    'rainbow',
    'cloud',
    'cable_car',
    'chess',
    'boat_part_ot',
    'grapefruit',
    'crow',
    'bracelet',
    'air_traffic_ot',
    'lily',
    'donkey',
    'cowboy_hat',
    'crown',
    'lavender',
    'dustpan',
    'jersey',
    'bolt',
    'sofa_bed',
    'sports_musical_ot',
    'cradle',
    'maple',
    'porcupine',
    'sleeping-bag',
    'shorts',
    'hockey',
    'willow',
    'shotgun',
    'wax_gourd',
    'rearview_mirror',
)

assert len(CLASSES) == len(_CLASSES)

@DATASETS.register_module()
class EntitySegDataset(CocoDataset):
    METAINFO = {'classes': CLASSES, 'thing_classes': THING_CLASSES, 'stuff_classes': STUFF_CLASSES}
