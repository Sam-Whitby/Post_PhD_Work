/*
  Copyright (c) 2015-2016 Lester Hedges <lester.hedges+vmmc@gmail.com>

  This program is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  This program is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#include <cstdlib>
#include <iostream>
#include <cmath>

#include "Box.h"

extern double TOL;


Box::Box(const std::vector<double>& boxSize_, bool isLattice_) :
    boxSize(boxSize_), isLattice(isLattice_)
{
    dimension = boxSize.size();

    // Check dimensionality is valid.
    if (dimension != 2 && dimension != 3)
    {
        std::cerr << "[ERROR] Box: Invalid dimensionality!\n";
        exit(EXIT_FAILURE);
    }

    isPeriodic.resize(dimension);
    posMinImage.resize(dimension);
    negMinImage.resize(dimension);

    for (int i=0;i<dimension;i++)
    {
        isPeriodic[i] = true;
        posMinImage[i] = 0.5*boxSize[i];
        negMinImage[i] = -0.5*boxSize[i];
    }

    // ensure box length is integer
    if(isLattice && std::abs(round(boxSize[0])-boxSize[0]) > TOL) {  
        std::cerr << "[ERROR] Box: BoxLength must be an integer when on a lattice!\n";
        exit(EXIT_FAILURE);
    } 
}


Box::Box(const std::vector<double>& boxSize_, const std::vector<bool>& isPeriodic_) :
    boxSize(boxSize_),
    isPeriodic(isPeriodic_)
{
    dimension = boxSize.size();

    // Check dimensionality is valid.
    if (dimension != 2 && dimension != 3)
    {
        std::cerr << "[ERROR] Box: Invalid dimensionality!\n";
        exit(EXIT_FAILURE);
    }

    posMinImage.resize(dimension);
    negMinImage.resize(dimension);

    for (unsigned int i=0;i<dimension;i++)
    {
        posMinImage[i] = 0.5*boxSize[i];
        negMinImage[i] = -0.5*boxSize[i];
    }
}


void Box::periodicBoundaries(std::vector<double>& coord)
{
    if(!isLattice) {
        for (int i=0;i<dimension;i++) {
            if (coord[i] < 0) {
                coord[i] += boxSize[i];
            }
            else {
                if (coord[i] >= boxSize[i])  // TODO: add in a small epsilon, if on a lattice
                {
                    coord[i] -= boxSize[i];
                }
            }
        }
    }
    if(isLattice) {
        for (int i=0;i<dimension;i++) {
            coord[i] = round(coord[i]);
            if (coord[i] < 0) {
                coord[i] += boxSize[i];
            }
            else {
                if (coord[i] >= boxSize[i])  
                {
                    coord[i] -= boxSize[i];
                }
            }
        }
    }    
}

void Box::minimumImage(std::vector<double>& separation)
{
    for (unsigned int i=0;i<dimension;i++)
    {
        if (separation[i] < negMinImage[i])
        {
            separation[i] += isPeriodic[i]*boxSize[i];
        }
        else
        {
            if (separation[i] >= posMinImage[i])
            {
                separation[i] -= isPeriodic[i]*boxSize[i];
            }
        }
    }
}
