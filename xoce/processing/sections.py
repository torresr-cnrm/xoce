"""
"""

import numpy  as np
import xarray as xr

from ..api.generic              import XoceObject, set_attrs
from ..calc.formulas.constants  import CONST


class NEMOGridCutter(XoceObject):
    """
    """
    _Parameters = {
        "fmesh_coords": {'type': dict,
                        'default': dict()},
        "section_pts": {'type': list,
                        'default': list()},
        "variables": {'type': dict,
                      'default': dict()},
        "optim": {'type': bool, 
                   'default': True},
    }

    def __init__(self, dataset=None, **kargs):
        XoceObject.__init__(self, dataset)

        # store intermediate section variables during processing
        self._intersections = None
        self._list_pts_ij   = list()
        
        # add default processing parameter
        self._set_default_parameters(**kargs)


    def _get_point_index(self, point):
        err = ( (point[0] - self.fmesh_coords['lon'])**2. + (point[1] - self.fmesh_coords['lat'])**2. )**0.5
        ind = np.unravel_index(np.argmin(err.data), err.shape)
        
        return ind[::-1]

    def _get_distance_between_points(self, points, e_r=CONST.radius):
        if type(points) in [list, tuple]:
            points = np.array(points)
            
        dx     = (points[1,0]-points[0,0])*(np.pi/180)*e_r*np.cos(0.5*(points[1,1]-points[0,1])*np.pi/180)
        dy     = (points[1,1]-points[0,1])*(np.pi/180)*e_r
        dist   = np.sqrt(dx**2 + dy**2)

        return dist


    @set_attrs
    def execute(self):
        ds = self.dataset

        section_pts = self.section_pts
        mesh_coords = self.fmesh_coords

        # Check attributes compatibility
        for var in self.variables:
            if var not in ds.variables:
                raise Exception("Section cut error: '{}' ".format(var) + 
                        "is not in dataset variables ()".format(list(ds.variables)))

        if not set(('lon', 'lat'))  <= set(mesh_coords.keys()):
            raise Exception("Section cut error: 'lon', 'lat' coordinates not found in " + 
                        "fmesh_coords attribute.")
    
        # ----------------------------------------------------------------------------------
        # Compute section intersections with circles of latitude on the grid
        intersections = None
        section_parts = dict()

        ipt = 0
        while ipt < len(section_pts)-1:
            pt1_i = section_pts[ipt]
            pt2_i = section_pts[ipt+1]

            # get slope - note that computation of slp_i must be after its sign to detect change 
            if pt2_i[0] == pt1_i[0]:
                slp_sign = np.sign(np.inf)
            else:
                slp_sign = np.sign((pt2_i[1] - pt1_i[1]) / (pt2_i[0] - pt1_i[0]))
            
            if intersections is not None and np.sign(slp_i) != slp_sign:
                raise Exception("Section cut error: section defined by different slope signs detected.")

            if pt2_i[0] == pt1_i[0]:
                slp_i = np.inf
            else:
                slp_i  = (pt2_i[1] - pt1_i[1]) / (pt2_i[0] - pt1_i[0])

            # select a box within the two points
            lon_min = min(pt1_i[0], pt2_i[0])
            lon_max = max(pt1_i[0], pt2_i[0])
            lat_min = min(pt1_i[1], pt2_i[1])
            lat_max = max(pt1_i[1], pt2_i[1])

            conds =         (mesh_coords['lon'] >= lon_min) & (mesh_coords['lon'] <= lon_max)
            conds = conds & (mesh_coords['lat'] >= lat_min) & (mesh_coords['lat'] <= lat_max)

            lats  = np.unique(mesh_coords['lat'].where(conds, drop=True), axis=1)
            lats  = lats[np.squeeze(np.argsort( (lats - (pt1_i[1]))**2, axis=0))]

            intersecs       = np.tile(lats, 2)
            intersecs[:, 0] = pt1_i[0] + (intersecs[:,1] - pt1_i[1]) / slp_i

            section_parts[ipt] = len(lats)

            if intersections is not None:
                intersections = np.concat( (intersections, intersecs), axis=0)
            else:
                intersections = intersecs

            ipt += 1
        
        self._intersections = intersections
        
        # ----------------------------------------------------------------------------------
        # Find the serie of mesh's points that form the section
        list_pts_ij = [self._get_point_index(intersections[0])]

        while (list_pts_ij[-1] != self._get_point_index(intersections[-1])):
            i, j  = list_pts_ij[-1]
            
            point = [float(mesh_coords['lon'][j,i]), float(mesh_coords['lat'][j,i])]
            dist  = self._get_distance_between_points((point,  intersections[-1]))

            west  = [float(mesh_coords['lon'][j,i-1]), float(mesh_coords['lat'][j,i-1])]
            north = [float(mesh_coords['lon'][j+1,i]), float(mesh_coords['lat'][j+1,i])]
            east  = [float(mesh_coords['lon'][j,i+1]), float(mesh_coords['lat'][j,i+1])]
            south = [float(mesh_coords['lon'][j-1,i]), float(mesh_coords['lat'][j-1,i])]
            
            zmin = np.inf
            for adj in [west, north, east, south]:
                zdist = np.min([self._get_distance_between_points([adj, intersec]) for intersec in intersections])
                
                if self._get_distance_between_points((adj,  intersections[-1])) < dist and zdist < zmin:
                    newpt = adj
                    zmin = zdist

            list_pts_ij.append(self._get_point_index(newpt))

        self._list_pts_ij = list_pts_ij

        # ----------------------------------------------------------------------------------
        # Create section dataset from the serie of waypoints with appropriate coordinates
        list_lons  = np.array([float(mesh_coords['lon'][ij[1], ij[0]]) for ij in list_pts_ij])
        list_lats  = np.array([float(mesh_coords['lat'][ij[1], ij[0]]) for ij in list_pts_ij])

        coords     = {'longitude': xr.DataArray( data=(list_lons[:-1] + list_lons[1:])/2, dims=('seg')),  
                      'latitude':  xr.DataArray( data=(list_lats[:-1] + list_lats[1:])/2, dims=('seg')),
                     }
        
        section = xr.Dataset(coords=coords)

        # Get x and y indices from the serie of waypoints
        index_x  = np.array(list_pts_ij)[:,0]
        index_y  = np.array(list_pts_ij)[:,1]

        ix0      = index_x[np.argmin(index_x)]
        iy0      = index_y[np.argmin(index_y)]

        indx     = np.abs(index_x - ix0)
        indy     = np.abs(index_y - iy0)


        # ----------------------------------------------------------------------------------
        # Loop over variables to fill the section dataset (values are selected to be at the
        # center of each segment)
        #
        for var in self.variables:
            grid = self.variables[var]

            if grid not in ['U', 'V', 'W', 'T', 'F']:
                print("Section cut warning: grid '{}' for variable '{}' ".format(grid, var) + 
                      "unknown. Variable is skipped")
            
            da = ds[var]

            if set(('x', 'y')) <= set(da.dims):

                # select only required indices plus an extra one to do interpolation,
                # -- can speed up the algorithm
                iselx = np.append(index_x, np.max(index_x) + 1) 
                isely = np.append(index_y, np.max(index_y) + 1) 

                da  = da.isel({'x': np.unique(iselx), 'y': np.unique(isely)})

                if self.optim:
                    da  = da.load()

                # initialize arrays and loop index
                k   = 0
                a   = np.ones(da.isel({'x': 0, 'y': 0}).shape) * np.nan
                a   = np.expand_dims(a, axis=-1)
                a   = np.repeat(a, len(indx)-1, axis=-1)

                if 'lgthcello' not in section.variables:
                    lgth = np.ones(len(indx)-1) * np.nan
                
                # loop over each segment defined by two consecutive waypoints
                while k < len(indx)-1:

                    if indx[k] - indx[k+1] != 0:    # -- horizontal section 
                        iy    = indy[k]
                        ix    = max(indx[k], indx[k+1])
                        
                        signu = 0
                        signv = -1 * np.sign(slp_i)
                        if slp_i  ==  np.inf or slp_i == 0:
                            signv  =  1

                    else:                           # -- vertical section
                        ix = indx[k]
                        iy = max(indy[k], indy[k+1])

                        signu = 1
                        signv = 0

                    # fill array with data, that is the value at the center of the segment
                    if grid == 'U':
                        a[...,k] = signu * da.isel({'x': ix, 'y': iy}).data
                    elif grid == 'V':
                        a[...,k] = signv * da.isel({'x': ix, 'y': iy}).data
                    elif grid in ['W', 'T']:
                        a[...,k] = signu * 0.5 * (da.isel({'x': ix, 'y': iy}) + da.isel({'x': ix+1, 'y': iy  })).data \
                                 + signv * 0.5 * (da.isel({'x': ix, 'y': iy}) + da.isel({'x': ix,   'y': iy+1})).data
                    elif grid == 'F':
                        a[...,k] = signv * 0.5 * (da.isel({'x': ix, 'y': iy}) + da.isel({'x': ix+1, 'y': iy  })).data \
                                 + signu * 0.5 * (da.isel({'x': ix, 'y': iy}) + da.isel({'x': ix,   'y': iy+1})).data      

                    if 'lgthcello' not in section.variables and set(('e1v', 'e2u')) <= set(ds.variables):
                        lgth[...,k] = np.abs(signv) * ds['e1v'].isel({'x': ix0+ix, 'y': iy0+iy}).data \
                                    +        signu  * ds['e2u'].isel({'x': ix0+ix, 'y': iy0+iy}).data

                    k += 1

                # get new coordinates
                new_coords   = dict(da.coords)
                
                if 'latitude' in new_coords:
                    del new_coords['latitude']
                if 'longitude' in new_coords:
                    del new_coords['longitude']

                new_coords['latitude']  = coords['latitude']
                new_coords['longitude'] = coords['longitude']

                new_dims = tuple(list(da.isel({'x': 0, 'y': 0}).dims) + ['seg'])

                # add array to dataset
                section[var] = xr.DataArray(data=a, dims=new_dims, coords=new_coords)
                
                if 'lgthcello' not in section.variables:
                    section['lgthcello']  = xr.DataArray(data=lgth, dims=('seg'),  coords=coords    )


        return section


