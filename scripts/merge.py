import os
import sys

import onnx
import onnx.compose

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

feature_extractor = onnx.load('artifacts/components/modified_feature_extractor_rn.onnx')
cv_builder = onnx.load('artifacts/components/modified_cv_builder_rn.onnx')
cv_regulator = onnx.load('artifacts/components/modified_cv_regulator_rn.onnx')
dist_regressor = onnx.load('artifacts/components/modified_dist_regressor_rn.onnx')

# feature_extractor = onnx.compose.add_prefix(
#     feature_extractor,
#     'feature_extractor',
#     rename_edges = True,
#     rename_inputs = False,
#     rename_outputs = False,
#     rename_initializers = False,
#     rename_value_infos = False,
#     inplace = True,
# )

# onnx.save(feature_extractor, 'artifacts/components/feature_extractor_rn.onnx')

# cv_builder = onnx.compose.add_prefix(
#     cv_builder,
#     'cv_builder',
#     rename_edges = True,
#     rename_inputs = False,
#     rename_outputs = False,
#     rename_initializers = False,
#     rename_value_infos = False,
#     inplace = True,
# )

# onnx.save(cv_builder, 'artifacts/components/cv_builder_rn.onnx')

# cv_regulator = onnx.compose.add_prefix(
#     cv_regulator,
#     'cv_regulator',
#     rename_edges = True,
#     rename_inputs = False,
#     rename_outputs = False,
#     rename_initializers = False,
#     rename_value_infos = False,
#     inplace = True,
# )

# onnx.save(cv_regulator, 'artifacts/components/cv_regulator_rn.onnx')

# dist_regressor = onnx.compose.add_prefix(
#     dist_regressor,
#     'dist_regressor',
#     rename_edges = True,
#     rename_inputs = False,
#     rename_outputs = False,
#     rename_initializers = False,
#     rename_value_infos = False,
#     inplace = True,
# )

# onnx.save(dist_regressor, 'artifacts/components/dist_regressor_rn.onnx')

model = onnx.compose.merge_models(
    feature_extractor,
    cv_builder,
    io_map = [
        ('feats', 'feats'),
    ],
    inputs = ['imgs', 'grids', 'masks'],
    outputs = ['feats', 'vol'],
)

model = onnx.compose.merge_models(
    model,
    cv_regulator,
    io_map = [
        ('vol', 'vol')
    ],
    inputs = ['imgs', 'grids', 'masks'],
    outputs = [
        'feats',
        'vol',
        'costs',
        'cv_regulator/down_blks.0/blks/blks.2/Pad_output_0',
        'cv_regulator/down_blks.1/blks/blks.2/Pad_output_0',
        'cv_regulator/down_blks.2/blks/blks.2/Pad_output_0',
    ],
)

model = onnx.compose.merge_models(
    model,
    dist_regressor,
    io_map = [
        ('costs', 'costs')
    ],
    inputs = ['imgs', 'grids', 'masks'],
    outputs = [
        'feats',
        'vol',
        'costs',
        'inv_dist',
        'cv_regulator/down_blks.0/blks/blks.2/Pad_output_0',
        'cv_regulator/down_blks.1/blks/blks.2/Pad_output_0',
        'cv_regulator/down_blks.2/blks/blks.2/Pad_output_0',
    ],
)

onnx.save(model, 'artifacts/model_merged.onnx')

print('Done.')