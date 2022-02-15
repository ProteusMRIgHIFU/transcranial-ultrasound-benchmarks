function CalculateKGridArrayBowl(inHFile)
% Calculate weighted array from kArray (k-Wave) for spherical bowl source
% Function to be called from Python notebook that saves in file the spatial
% information for the source


N1=h5read(inHFile,'/N1');
N2=h5read(inHFile,'/N2');
N3=h5read(inHFile,'/N3');
dx=h5read(inHFile,'/SpatialStep');
BLITolerance=double(h5read(inHFile,'/BLITolerance'));
UpsamplingRate=double(h5read(inHFile,'/UpsamplingRate'));
bDisplay=h5read(inHFile,'/bDisplay');
SourceLocatonGridPoints=h5read(inHFile,'/SourceLocatonGridPoints');
source_roc = h5read(inHFile,'/TxRadius');
source_diameter = h5read(inHFile,'/TxDiameter');

% create the computational grid
kgrid = kWaveGrid(double(N1), dx, double(N2), dx, double(N3), dx);

% --------------------
% SOURCE
% --------------------
% create empty kWaveArray
karray = kWaveArray('BLITolerance', BLITolerance, 'UpsamplingRate', UpsamplingRate);


bowl_pos = [kgrid.x_vec(SourceLocatonGridPoints(1)), kgrid.y_vec(SourceLocatonGridPoints(2)), kgrid.z_vec(SourceLocatonGridPoints(3))];
focus_pos = [kgrid.x_vec(SourceLocatonGridPoints(1)), kgrid.y_vec(SourceLocatonGridPoints(2)), kgrid.z_vec(end)];

% add disc shaped element at one end of the grid
karray.addBowlElement(bowl_pos, source_roc, source_diameter, focus_pos);
    
outFile=split(inHFile,'-kArrayIn.h5');
outFile=[outFile{1},'-kArrayOut.mat'];


% assign binary mask
source_weights = karray.getElementGridWeights(kgrid, 1);
save(outFile,'source_weights','-v7.3');

if bDisplay>0
    source.p_mask=source_weights~=0;
    figure;subplot(1,2,1);imagesc(squeeze(sum(source.p_mask*1.0,1)));colorbar;daspect([1,1,1]);title('YZ projection');subplot(1,2,2);imagesc(squeeze(sum(source.p_mask*1.0,3)));daspect([1,1,1]);colorbar;title('XZ projection');sgtitle('source MASK projection')
end

