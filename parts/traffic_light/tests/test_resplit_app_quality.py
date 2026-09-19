import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from parts.traffic_light.tools.app_quality_common import read_json, rows, sha256, write_json
from parts.traffic_light.tools.resplit_app_quality import allocate, grouped_indices, label_features, prepare, sequence_key


class ResplitTests(unittest.TestCase):
    def test_sequence_names(self):
        self.assertEqual(sequence_key('/x/MP_KSC_P000101.jpg', 100), 'MP_KSC_P:1')
        self.assertEqual(sequence_key('/x/MP_SEL_000199.jpg', 100), 'MP_SEL_:1')

    def test_duplicate_content_unites_sequence_groups(self):
        records = [{'source_image': f'img_{i:06d}.jpg', 'source_sha256': h}
                   for i,h in [(1,'a'), (2,'b'), (500,'a'), (501,'c'), (900,'d')]]
        groups=grouped_indices(records,100)
        self.assertEqual(sorted(map(len,groups)),[1,4])
        with self.assertRaises(ValueError):
            grouped_indices(records,0)

    def test_label_sizes_in_decoded_image(self):
        counts,sizes=label_features('0 .5 .5 .01 .01\n0 .5 .5 .04 .04\n1 .5 .5 .8 .2\n',(960,540))
        self.assertEqual(dict(counts),{'pedestrian_signal':2,'crosswalk':1})
        self.assertEqual(dict(sizes),{'tiny':1,'small':1})
        with self.assertRaises(ValueError):
            label_features('0 nan .5 .1 .1',(960,540))

    def test_allocation_deterministic_and_balanced(self):
        records=[{'split':['train','val','test'][i%3], 'class_counts':{'pedestrian_signal':2 if i%5==0 else 1},
                  'signal_size_counts':{'small':1}} for i in range(100)]
        groups=[[i] for i in range(100)]
        result=allocate(records,groups,[.8,.1,.1],42)
        self.assertEqual(result,allocate(records,groups,[.8,.1,.1],42))
        self.assertEqual(set(result),set(range(100)))
        for split,target in [('train',80),('val',10),('test',10)]:
            self.assertLessEqual(abs(sum(s==split for s,g in result.values())-target),2)
        with self.assertRaises(ValueError):
            allocate(records,groups,[.8,.2,.2],42)

    def test_prepare_preserves_bytes_labels_and_crop_membership(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp); old=root/'old'; det=old/'detector_app'; crop=old/'classifier_app'
            det.mkdir(parents=True); crop.mkdir()
            records=[]; sources=[]; crops=[]
            for i in range(20):
                split=['train','val','test'][i%3]
                source=f'/source/frame_{i*100:06d}.jpg'
                image=f'images/{split}/{i}.jpg'; label=f'labels/{split}/{i}.txt'; cr=f'{split}/red/{i}.png'
                for path in (det/image,det/label,crop/cr):path.parent.mkdir(parents=True,exist_ok=True)
                # Byte fixtures: preparation links existing files and does not decode/re-encode them.
                (det/image).write_bytes(f'jpeg-{i}'.encode()); (crop/cr).write_bytes(f'crop-{i}'.encode())
                (det/label).write_text('0 .5 .5 .01 .02\n1 .5 .5 .8 .2\n')
                digest=sha256(det/image)
                records.append(dict(source_image=source,source_sha256=digest,split=split,image=image,label=label,label_sha256=sha256(det/label)))
                sources.append(dict(source_image=source,source_sha256=digest,transport_sha256=digest,size=[960,540]))
                crops.append(dict(source_image=source,split=split,crop=cr,class_name='red'))
            for path,data in [(det/'manifest.jsonl',records),(old/'sources.jsonl',sources),(crop/'manifest.jsonl',crops)]:
                path.write_text(''.join(json.dumps(r)+'\n' for r in data))
            write_json(old/'summary.json',dict(complete=True,sources_sha256=sha256(old/'sources.jsonl')))
            write_json(root/'base.json',dict(prepared_root=str(old)))
            config=dict(base_config=str(root/'base.json'),base_config_sha256=sha256(root/'base.json'),
                        prepared_manifest_sha256=sha256(det/'manifest.jsonl'),output=str(root/'new'),sequence_block_size=100,
                        ratios=[.8,.1,.1],seed=42)
            with contextlib.redirect_stdout(io.StringIO()):prepare(config)
            new=root/'new'; membership={}
            previous={r['source_image']:r for r in records}
            for row in rows(new/'detector_app/manifest.jsonl'):
                original=previous[row['source_image']]
                self.assertEqual((new/'detector_app'/row['image']).read_bytes(),(det/original['image']).read_bytes())
                self.assertEqual((new/'detector_app'/row['label']).read_bytes(),(det/original['label']).read_bytes())
                self.assertTrue((new/'detector_app'/row['image']).is_symlink())
                self.assertFalse((new/'detector_app'/row['label']).is_symlink())
                membership[row['source_image']]=row['split']
            for row in rows(new/'classifier_app/manifest.jsonl'):
                self.assertEqual(row['split'],membership[row['source_image']])
                self.assertTrue((new/'classifier_app'/row['crop']).is_file())
            self.assertTrue(read_json(new/'summary.json')['complete'])
            self.assertEqual(sha256(det/'manifest.jsonl'),config['prepared_manifest_sha256'])
            with self.assertRaises(FileExistsError),contextlib.redirect_stdout(io.StringIO()):prepare(config)


if __name__=='__main__':
    unittest.main()
