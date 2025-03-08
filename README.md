# HeightAware-BEV

A simple BEV perception framework which achieves efficient and accurate view transformation through
height-aware feature mapping.



<!-- <p align="center">
  <h2 align="center">HeightAware-BEV: Height-Aware Feature Mapping for Efficient Bird’s-Eye-View Perception</h2>
</p> -->

⚠️ **Notice: This repository is under review. Full documentation and data will be released upon paper acceptance.**

## Demo
<img src='./assets/demo.gif'>

## Main results
<div align="center">
<table border="1">
  <caption><i><b>State-of-the-art Comparison:</b> Vehicle segmentation of various static models at 448x800 image resolution with visibility filtering. More details can be found in our paper.</i></caption>
    <tr>
        <th>Models</th>
        <th>Publication</th>
        <th>backbone</th>
        <th>IOU</th>
        <th>FPS</th>
    </tr>
    <tr class="highlight-column">
        <td><a href="https://arxiv.org/abs/2205.02833">CVT</a></td>
        <td>CVPR2022</td>
        <td>ENb4</td>
        <td>37.7</td>
        <td></td>
    </tr>
    </tr>
    <tr class="highlight-column">
        <td><a href="https://arxiv.org/abs/2203.17270">BEVFormer</a></td>
        <td>ECCV2022</td>
        <td>RN50</td>
        <td>45.5</td>
        <td></td>
    </tr>
    </tr>
    <tr class="highlight-column">
        <td><a href="https://arxiv.org/abs/2206.07959">SimpleBeV</a></td>
        <td>ICRA2023</td>
        <td>RN50</td>
        <td>46.6</td>
        <td>65FPS A100</td>
    </tr>
    </tr>
    <tr class="highlight-column">
        <td><a href="https://arxiv.org/abs/2312.00703">PointBeV</a></td>
        <td>CVPR2024</td>
        <td>ENb4</td>
        <td>47.6</td>
        <td>15FPS A100</td>
    </tr>
    </tr>
    <tr class="highlight-column">
        <td>Ours</td>
        <td>under review</td>
        <td>RN50</td>
        <td>47.8</td>
        <td>63FPS A40<br>60FPS 2080TI</td>
    </tr>
</table>
</div>

## News

## Setup

## Traing

## Evaluation

## Checkpoints

## Todo





